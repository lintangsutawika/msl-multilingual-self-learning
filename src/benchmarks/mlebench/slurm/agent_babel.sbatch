#!/usr/bin/env bash
# MLE-bench tasks with an LLM agent on Babel, one task per GPU node: vLLM on the GPU, the task's attempts in
# Singularity (run_babel.sh: public data mounted read-only, `validate` graded on the host). The agent reaches vLLM
# directly on 127.0.0.1 (Singularity shares the host network), so no relay is needed.
#
# Agent and sampling settings are the training rollouts' (scripts/rollout.sh): the model's default sampling (no
# temperature/top_p set), at most MAX_GEN_TOKENS per response (the relay's cap, here vLLM's default max_tokens: the
# agent sends no limit), MAX_ITERATIONS steps, OpenHands' condenser past CONDENSER_MAX_EVENTS events, no terminal
# action longer than TERMINAL_MAX_TIMEOUT s, and the task's agent timeout (4 h) as the trial timeout.
# Resources: each attempt as in the MLE-Dojo sweep (4 CPUs, 16 GB; Babel task builds set it), so 4 attempts need
# 16 CPUs and 64 GB plus vLLM's host memory: --cpus-per-task=16 --mem=96G. The agent gets no GPU (the GPU serves vLLM).
# Each trial installs the agent (uv, Python, OpenHands) from the internet; a trial whose download fails
# (NetworkConnectionError, before the agent starts) is retried up to 3 times. Nothing else is retried.
#
#   # one task:
#   sbatch --export=ALL,TASKS=/path/task src/benchmarks/mlebench/slurm/agent_babel.sbatch
#   # every task of a dir, one array element each. Preempted elements requeue and resume by themselves; after a
#   # cancel, time limit or crash, resubmit the same command: finished attempts are kept, done tasks skipped:
#   T=/path/to/tasks/mlebench-low
#   sbatch --array=0-$(($(ls -d $T/*/ | wc -l) - 1)) --export=ALL,TASKS_DIR=$T,SWEEP_NAME=qwen3.5-9b-k4 \
#       src/benchmarks/mlebench/slurm/agent_babel.sbatch
#
# Env: TASKS (task dirs, space-separated) or TASKS_DIR (+ SLURM_ARRAY_TASK_ID picks one, in `ls` order: don't add or
#      remove task dirs there while the array runs, or a requeued element picks another task);
#      MODEL (default Qwen/Qwen3.5-9B); N_ATTEMPTS (4, run at once); SWEEP_NAME (default agent; jobs go to
#      <task parent>/../sweeps/<SWEEP_NAME>); MAX_GEN_TOKENS (16384), MAX_ITERATIONS (200),
#      CONDENSER_MAX_EVENTS (120), TERMINAL_MAX_TIMEOUT (1800): as in scripts/rollout.sh;
#      HARBOR_EXTRA_ARGS (extra `harbor run` flags, e.g. --agent-timeout-multiplier 2 for 8 h attempts);
#      VLLM_CUDA_SIF, CACHE_DIR, GRADER_PY: site paths, from src/benchmarks/mlebench/site.env (see site.env.example).
# Submit from the repo root (logs go to logs/). --account/--partition/--qos on the sbatch command line override the
# defaults below.
#SBATCH --job-name=mlebench-agent
#SBATCH --account=teruko
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --constraint=VRAM_48GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out
set -uo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
cd "$REPO_ROOT"
if [ -n "${TASKS_DIR:-}" ]; then
  mapfile -t ALL < <(ls -d "${TASKS_DIR%/}"/*/ | sed 's:/$::')
  TASKS="${ALL[${SLURM_ARRAY_TASK_ID:?TASKS_DIR needs an array job}]}"
fi
: "${TASKS:?set TASKS or TASKS_DIR}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
SWEEP_NAME="${SWEEP_NAME:-agent}"
N_ATTEMPTS="${N_ATTEMPTS:-4}"
MAX_GEN_TOKENS="${MAX_GEN_TOKENS:-16384}"
TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$$}}_${SLURM_ARRAY_TASK_ID:-0}"
PORT="${PORT:-$((20000 + $(printf '%s' "$TAG" | cksum | cut -d' ' -f1) % 20000))}"
[ -f src/benchmarks/mlebench/site.env ] && . src/benchmarks/mlebench/site.env  # site paths (site.env.example)
: "${VLLM_CUDA_SIF:?set VLLM_CUDA_SIF (src/benchmarks/mlebench/site.env.example)}" "${CACHE_DIR:?set CACHE_DIR}"
export VLLM_CUDA_SIF CACHE_DIR
FIRST=${TASKS%% *}
export JOBS_DIR="${JOBS_DIR:-$(dirname "$(dirname "$FIRST")")/sweeps/${SWEEP_NAME}}"
mkdir -p "$JOBS_DIR"
log() { echo "[agent $(date '+%m-%d %H:%M:%S')] $*"; }

# Preemption-safe: each task is one Harbor job with a fixed name (<SWEEP_NAME>-<task>). Slurm requeues a preempted job
# and run_babel.sh (RESUME=1) resumes it: finished attempts are kept, only the cut-off attempts rerun (from scratch:
# an attempt's agent state can't be restored). A task is done once all N_ATTEMPTS attempts have a result that is not
# an interruption (CancelledError) or a failed agent download (NetworkConnectionError).
todo=()
for t in $TASKS; do
  name=$(basename "$t")
  j="$JOBS_DIR/${SWEEP_NAME}-${name}"
  n=0
  # by exception type (a timed-out attempt's traceback also mentions CancelledError, so no grep)
  [ -d "$j" ] && n=$(python3 - "$j" <<'EOF'
import glob, json, sys
n = 0
for p in glob.glob(sys.argv[1] + "/*/result.json"):
    e = (json.load(open(p)).get("exception_info") or {}).get("exception_type")
    n += e not in ("CancelledError", "NetworkConnectionError")
print(n)
EOF
)
  if [ "$n" -ge "$N_ATTEMPTS" ]; then log "skip $name: all ${N_ATTEMPTS} attempts done"; else todo+=("$t"); [ "$n" -gt 0 ] && log "$name: resuming (${n}/${N_ATTEMPTS} done)"; fi
done
[ ${#todo[@]} -eq 0 ] && { log "nothing to do"; exit 0; }

VSCRATCH="/scratch/${USER}/vllm-${TAG}"
mkdir -p "$VSCRATCH"
log "host=$(hostname) model=${MODEL} port=${PORT} tasks=${todo[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# The model's serve flags (configs/serve/<MODEL>.json) plus the response cap as vLLM's default max_tokens.
SERVE_ARGS=$(python3 -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["extra_args"]))' "configs/serve/${MODEL}.json")
VLLM_ARGS_EXTRA="${SERVE_ARGS} --override-generation-config {\"max_new_tokens\":${MAX_GEN_TOKENS}}" \
DATA_PARALLEL=1 TENSOR_PARALLEL=1 PORT="$PORT" BASE_DIR="$VSCRATCH" MODEL="$MODEL" \
  bash scripts/serve/serve_vllm.sh > "${JOBS_DIR}/vllm-${TAG}.log" 2>&1 &
VLLM=$!
trap 'pkill -TERM -P $VLLM 2>/dev/null; kill $VLLM 2>/dev/null; rm -rf "$VSCRATCH"' EXIT
trap 'exit 143' TERM INT
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  kill -0 $VLLM 2>/dev/null || { log "vLLM exited during startup"; tail -30 "${JOBS_DIR}/vllm-${TAG}.log"; exit 1; }
  sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || { log "vLLM not healthy after 30 min"; exit 1; }
log "vLLM healthy; $(grep -o "Default sampling parameters.*" "${JOBS_DIR}/vllm-${TAG}.log" | head -1)"

RESUME=1 RESUME_LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" RESUME_LLM_API_KEY=EMPTY \
AGENT="${AGENT:-src.rollout.openhands_agent:CondensingOpenHandsSDK}" JOB_PREFIX="$SWEEP_NAME" N_ATTEMPTS="$N_ATTEMPTS" \
AGENT_ARGS="-m litellm_proxy/${MODEL} --ae LLM_API_KEY=EMPTY --ae LLM_BASE_URL=http://127.0.0.1:${PORT}/v1
  --ae LLM_MODEL=litellm_proxy/${MODEL} --ae MAX_ITERATIONS=${MAX_ITERATIONS:-200}
  --ae CONDENSER_MAX_EVENTS=${CONDENSER_MAX_EVENTS:-120} --ae TERMINAL_MAX_TIMEOUT=${TERMINAL_MAX_TIMEOUT:-1800}
  --max-retries 3 --retry-include NetworkConnectionError ${HARBOR_EXTRA_ARGS:-}" \
  src/benchmarks/mlebench/run_babel.sh "${todo[@]}"
log "finished rc=$?"
