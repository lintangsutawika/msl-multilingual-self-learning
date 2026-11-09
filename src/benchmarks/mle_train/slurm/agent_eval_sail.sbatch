#!/usr/bin/env bash
# Real-agent evaluation of MLE training tasks on SAIL, with the model served on Babel (one job, one GPU).
# Mirrors the msl rollout job (scripts/rollout.sh) on CMU Babel / Slurm:
#
#   openhands-sdk in a SAIL sandbox --https--> Modal relay /sessions/<trial>/v1 --ws--> node_forwarder --> vLLM
#                                                                                         |
#                                                                     RUN_DIR/spool/<trial>.jsonl (calls, signed
#                                                                     validate scores as the trainer would see them)
#
# SAIL sandboxes run outside CMU, so they reach vLLM only through the per-job Modal relay; the node dials out.
# The relay app is stopped when the job ends.
#
# Prerequisites (once): `sail auth login` (SAIL key in ~/.sail/auth.toml) and `.venv/bin/modal token new`
# (Modal token in ~/.modal.toml). Neither key is read or printed by this script.
#
# Submit from the repo root:
#   mkdir -p logs
#   TASKS=/data/user_data/jeffc2/datasets/mle-train/harbor EVAL_TASKS=playground-series-s3e5 \
#       sbatch --export=ALL src/benchmarks/mle_train/slurm/agent_eval_sail.sbatch
#
# Env: TASKS (Harbor tasks dir, required); EVAL_TASKS (comma-separated task names, required);
#      MODEL (default Qwen/Qwen3.5-9B); MAX_ITERATIONS (default 200); N_CONCURRENT (default 2);
#      RUN_DIR (default <TASKS>/../agent-eval/<job id>); VLLM_CUDA_SIF; CACHE_DIR (holds huggingface/).
#SBATCH --job-name=mle-agent-eval
#SBATCH --account=teruko
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --constraint=VRAM_48GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
cd "$REPO_ROOT"
: "${TASKS:?set TASKS to a directory of Harbor tasks}"
: "${EVAL_TASKS:?set EVAL_TASKS, e.g. playground-series-s3e5}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
MAX_ITERATIONS="${MAX_ITERATIONS:-200}"  # as in scripts/rollout.sh; 100 steps can run out in ~20 min
N_CONCURRENT="${N_CONCURRENT:-2}"
JOBTAG="${SLURM_JOB_ID:-local-$$}"
RUN_DIR="${RUN_DIR:-$(dirname "$TASKS")/agent-eval/${JOBTAG}}"
SPOOL_DIR="${RUN_DIR}/spool"
PORT="${PORT:-$((20000 + $(printf '%s' "$JOBTAG" | cksum | cut -d' ' -f1) % 20000))}"
export VLLM_CUDA_SIF="${VLLM_CUDA_SIF:-/data/user_data/jeffc2/containers/vllm-openai-v0.31.0.sif}"
export CACHE_DIR="${CACHE_DIR:-/data/user_data/jeffc2/.cache}"
SCRATCH="/scratch/${USER}/mle-agent-eval-${JOBTAG}"
RELAY_APP="mle-eval-relay-${JOBTAG}"
# The repo's modal client (1.5.x) can't build with a new workspace's default image builder (2026.10).
export MODAL_IMAGE_BUILDER_VERSION="${MODAL_IMAGE_BUILDER_VERSION:-2025.06}"
PY="${REPO_ROOT}/.venv/bin/python"
mkdir -p "$RUN_DIR" "$SPOOL_DIR" "$SCRATCH"
log() { echo "[agent-eval $(date +%H:%M:%S)] $*"; }
ATTEMPT_START="$(date +%s)"
log "job=${JOBTAG} host=$(hostname) model=${MODEL} tasks=${EVAL_TASKS} run_dir=${RUN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# `modal app stop` asks for confirmation without -y; in a batch job nobody answers and the relay keeps running.
stop_relay() {
  "${REPO_ROOT}/.venv/bin/modal" app stop -y "${RELAY_APP}" >/dev/null 2>&1 || true
}
PIDS=()
TPIDS=()
cleanup() {
  trap - EXIT TERM INT
  log "shutting down (stopping relay app ${RELAY_APP})"
  # Trials first: SIGTERM lets harbor tear down its SAIL sandbox.
  for p in "${TPIDS[@]}"; do pkill -TERM -P "$p" 2>/dev/null || true; kill "$p" 2>/dev/null || true; done
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
  stop_relay
  if "${REPO_ROOT}/.venv/bin/modal" app list --json 2>/dev/null | "$PY" -c \
      'import json,sys; sys.exit(not any(a.get("description")==sys.argv[1] and a.get("state")=="deployed" for a in json.load(sys.stdin)))' \
      "$RELAY_APP"; then
    log "WARNING: relay app ${RELAY_APP} still deployed; stop it: .venv/bin/modal app stop -y ${RELAY_APP}"
  fi
  rm -rf "$SCRATCH"
}
trap cleanup EXIT
trap 'exit 143' TERM INT
# A preempted attempt of this job (same job id, requeued) may have left its relay deployed.
[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ] && log "requeued (restart ${SLURM_RESTART_COUNT}); stopping the previous attempt's relay"
stop_relay

# 1. vLLM on this node's GPU (serve params from configs/serve/<MODEL>.json; 1 replica on 1 GPU).
DATA_PARALLEL=1 TENSOR_PARALLEL=1 PORT="$PORT" BASE_DIR="$SCRATCH" MODEL="$MODEL" \
  bash scripts/serve/serve_vllm.sh > "${RUN_DIR}/vllm.log" 2>&1 &
PIDS+=("$!")
log "vLLM starting on :${PORT}; waiting for health (log ${RUN_DIR}/vllm.log)"
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  kill -0 "${PIDS[0]}" 2>/dev/null || { log "vLLM exited during startup"; tail -30 "${RUN_DIR}/vllm.log"; exit 1; }
  sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || { log "vLLM not healthy after 30 min"; exit 1; }
log "vLLM healthy"

# 2. Per-job Modal relay with a fresh secret (the secret is the agents' API key; never logged).
RELAY_SECRET="$(openssl rand -hex 16)"
DEPLOY_OUT="$(RELAY_APP_NAME="$RELAY_APP" RELAY_SECRET_NAME="$RELAY_APP" RELAY_SECRET="$RELAY_SECRET" \
  "${REPO_ROOT}/.venv/bin/modal" deploy src/relay/public_api.py 2>&1)" || { echo "$DEPLOY_OUT"; log "relay deploy failed (modal token?)"; exit 1; }
RELAY_URL="$(printf '%s' "$DEPLOY_OUT" | grep -oE 'https://[a-zA-Z0-9._-]+\.modal\.run' | head -1)"
[ -n "$RELAY_URL" ] || { echo "$DEPLOY_OUT"; log "could not parse relay URL"; exit 1; }
log "relay deployed: ${RELAY_URL}"

# 3. Forwarder: bridge to vLLM + capture every chat completion into the spool; validate scores are checked
#    against each task's environment/score_key (SCORE_KEYS_DIR = the tasks dir).
RELAY_WS_URL="wss://${RELAY_URL#https://}/bridge" RELAY_SECRET="$RELAY_SECRET" \
  VLLM_LOCAL_URL="http://127.0.0.1:${PORT}" CAPTURE_DIR="$SPOOL_DIR" SCORE_KEYS_DIR="$TASKS" \
  "$PY" src/relay/node_forwarder.py > "${RUN_DIR}/forwarder.log" 2>&1 &
PIDS+=("$!")
for _ in $(seq 1 30); do
  curl -sf -H "Authorization: Bearer ${RELAY_SECRET}" "${RELAY_URL}/v1/models" >/dev/null 2>&1 && break
  sleep 5
done
curl -sf -H "Authorization: Bearer ${RELAY_SECRET}" "${RELAY_URL}/v1/models" >/dev/null \
  || { log "relay -> vLLM check failed"; tail -20 "${RUN_DIR}/forwarder.log"; exit 1; }
log "relay -> forwarder -> vLLM ok"

# 4. One trial per task: openhands-sdk on SAIL, LLM calls filed under the trial's relay session.
#    Trial ids start with the task name (the forwarder picks the score key from it).
run_trial() {
  local task="$1" trial="$1__$(date +%m%d%H%M%S)"
  log "start ${trial}"
  "${REPO_ROOT}/.venv/bin/harbor" run -p "${TASKS}/${task}" \
    -a openhands-sdk -m "litellm_proxy/${MODEL}" -e sail.harbor:SailboxEnvironment -n 1 \
    --ae "LLM_API_KEY=${RELAY_SECRET}" --ae "LLM_BASE_URL=${RELAY_URL}/sessions/${trial}/v1" \
    --ae "LLM_MODEL=litellm_proxy/${MODEL}" --ae "MAX_ITERATIONS=${MAX_ITERATIONS}" \
    --job-name "$trial" -o "${RUN_DIR}/jobs" -y --quiet > "${RUN_DIR}/${trial}.log" 2>&1 \
    && log "done ${trial}" || log "FAILED ${trial} (see ${RUN_DIR}/${trial}.log)"
}
IFS=',' read -ra TLIST <<< "$EVAL_TASKS"
for t in "${TLIST[@]}"; do
  [ -d "${TASKS}/${t}" ] || { log "skip ${t}: not in ${TASKS}"; continue; }
  while [ "$(jobs -rp | grep -cvxF -e "${PIDS[0]}" -e "${PIDS[1]}")" -ge "$N_CONCURRENT" ]; do sleep 15; done
  run_trial "$t" &
  TPIDS+=("$!")
done
for p in "${TPIDS[@]}"; do wait "$p" || true; done

# 5. Summary: verifier reward per trial + the validate scores the forwarder accepted (signed) or rejected.
#    "rejected" = score=... text without a valid signature, e.g. the agent quoting a score in its own code or notes.
"$PY" - "$RUN_DIR" "$ATTEMPT_START" <<'PY'
import glob, json, os, sys
run, since = sys.argv[1], float(sys.argv[2])
# A requeued job reuses RUN_DIR: report only this attempt's trials.
mine = lambda p: os.path.getmtime(p) >= since
for d in sorted(filter(mine, glob.glob(f"{run}/jobs/*/*__*/"))):
    trial = os.path.basename(d.rstrip("/"))
    reward = open(d + "verifier/reward.txt").read().strip() if os.path.exists(d + "verifier/reward.txt") else "-"
    final = open(d + "verifier/final_score.json").read().strip() if os.path.exists(d + "verifier/final_score.json") else ""
    print(f"{trial}: reward={reward} {final[:200]}")
for f in sorted(filter(mine, glob.glob(f"{run}/spool/*.jsonl"))):
    recs = [json.loads(l) for l in open(f) if l.strip()]
    acc = [s for r in recs for s in r.get("observed_validate_scores") or []]
    rej = [s for r in recs for s in r.get("rejected_validate_scores") or []]
    aligned = sum(1 for r in recs if r.get("ids_logprobs_aligned"))
    print(f"spool {os.path.basename(f)}: {len(recs)} calls ({aligned} with token ids+logprobs), "
          f"validate calls {sum(1 for r in recs if r.get('calls_validate'))}, signed scores {acc}, rejected {rej}")
PY
log "finished"
