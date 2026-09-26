#!/usr/bin/env bash
# Serve vLLM (from a CUDA SIF) on this node and run the LeetCode harbor eval
# against it on the SAME node. The agent runs in a Singularity container that
# shares the host network, so MODEL_BASE_URL=127.0.0.1:8000 is reachable from it.
#
# run_leetcode.sh only launches harbor; this wrapper brings up the model first,
# waits for /health, then delegates to run_leetcode.sh, and tears vLLM down on exit.
#
# PBS (ABCI) and SLURM both supported. ABCI: submit with qsub, pass resources via -v
# (PBS takes no positional script args). Example:
#
#   qsub -N leetcode-eval -j oe -o logs/ -P gai51740 -q R9920261300 \
#       -v RTYPE=rt_HF scripts/eval/leetcode_eval.sbatch
#
# Knobs (env vars):
#   VLLM_CUDA_SIF    CUDA vLLM Singularity image (required for local serve).
#   MODEL            bare HF repo to serve, e.g. Qwen/Qwen3.5-9B.
#   VLLM_PORT        serve port (default 8000).
#   TENSOR_PARALLEL  TP (default 1); DATA_PARALLEL DP (default 1).
#   MAX_MODEL_LEN    context length (default 131072).
#   VLLM_EXTRA_ARGS  extra vLLM flags; auto-loaded from configs/serve/<MODEL>.json
#                    when VLLM_SERVE_CONFIG_MODE is on (or SERVE_CONFIG given).
#   GPU_MEM_UTIL     gpu-memory-utilization (default 0.85).
#   CACHE_DIR        single host dir for ALL container caches, bound at /cache
#                    (HF models, vLLM compile cache, tmp). Default ${BASE_DIR}/.cache.
#   BASE_DIR         host base dir for container scratch/tmp (default ${PBS_LOCALDIR:-${TMPDIR:-/tmp}}).
#   N_CONCURRENT     passed to run_leetcode.sh (default 32).
#   TASKS            task/dataset path (default benchmarks/leetcode/tasks).
#   N_TASKS          subset limit for a smoke run; empty = all.
set -euo pipefail
# Resolve the repo root. Under qsub, BASH_SOURCE[0] is the spooled script copy and
# PBS_O_WORKDIR is the submit dir -- which is $HOME if qsub was run outside the repo,
# so trust it ONLY when it actually contains this project (scripts/eval/run.sh).
# Otherwise derive from the script's own location (works for interactive runs too).
# Walk up from the submit dir (or CWD) to the dir that actually contains this project
# (marked by scripts/eval/run.sh). This is robust regardless of where qsub was run from
# (PBS_O_WORKDIR is $HOME if qsub ran outside the repo) and of BASH_SOURCE[] being the
# spooled copy under qsub. An explicit REPO_ROOT env still wins.
resolve_repo_root() {
    local start
    start="${PBS_O_WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
    local dir="${start}"
    while [ -n "${dir}" ] && [ "${dir}" != "/" ]; do
        if [ -x "${dir}/scripts/eval/run.sh" ]; then
            echo "${dir}"; return
        fi
        dir="$(dirname "${dir}")"
    done
    # Last resort: known repo location for this user.
    [ -x "${HOME}/msl-multilingual-self-learning/scripts/eval/run.sh" ] \
        && echo "${HOME}/msl-multilingual-self-learning"
}
REPO_ROOT="${REPO_ROOT:-$(resolve_repo_root)}"
cd "${REPO_ROOT}"
if [ -n "${PBS_JOBID:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p logs
    exec > >(tee -a "logs/${PBS_JOBID}.out") 2>&1
    echo "[log] teeing to logs/${PBS_JOBID}.out (tail -f it)"
fi
[ -f .env ] && . .env

# --- knobs ----------------------------------------------------------------
VLLM_CUDA_SIF="${VLLM_CUDA_SIF:-}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
VLLM_PORT="${VLLM_PORT:-8000}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
DATA_PARALLEL="${DATA_PARALLEL:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
SERVE_CONFIG="${SERVE_CONFIG:-configs/serve/${MODEL}.json}"
BASE_DIR="${BASE_DIR:-${PBS_LOCALDIR:-${TMPDIR:-/tmp}}}"
CACHE_DIR="${CACHE_DIR:-${BASE_DIR}/.cache}"
N_CONCURRENT="${N_CONCURRENT:-32}"
TASKS="${TASKS:-benchmarks/leetcode/tasks}"
# Default job name: <dataset>--<model>, with any / in the model replaced by --.
DATASET_NAME="${TASKS////---}"
DEFAULT_JOB="${DATASET_NAME}--${MODEL//\//--}"
export JOB_NAME="${JOB_NAME:-${DEFAULT_JOB}}"

[ -n "${VLLM_CUDA_SIF}" ] || { echo "ERROR: VLLM_CUDA_SIF not set" >&2; exit 2; }
[ -f "${VLLM_CUDA_SIF}" ] || { echo "ERROR: VLLM_CUDA_SIF not found: ${VLLM_CUDA_SIF}" >&2; exit 2; }

APPTAINER_BIN="${APPTAINER_BIN:-$(command -v apptainer || command -v singularity || echo singularity)}"
SERVE_READY_TRIES="${SERVE_READY_TRIES:-180}"   # ~15 min at 5s

VLLM_PID=""
cleanup() {
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[cleanup] stopping vLLM (pid=${VLLM_PID})"
        kill "${VLLM_PID}" 2>/dev/null; wait "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT SIGTERM SIGINT

# --- serve vLLM (delegated to scripts/serve/serve_vllm.sh) ---------------------
# The serve script owns the per-model serve-arg loading (configs/serve/<MODEL>.json),
# node-local cache wiring, the apptainer exec, and backgrounding; it prints the vLLM
# PID on stdout. This batch owns readiness checks and cleanup (see VLLM_PID below).
export VLLM_CUDA_SIF MODEL VLLM_PORT TENSOR_PARALLEL DATA_PARALLEL \
       MAX_MODEL_LEN GPU_MEM_UTIL BASE_DIR CACHE_DIR SERVE_CONFIG \
       VLLM_EXTRA_ARGS CUDA_VISIBLE_DEVICES APPTAINER_BIN
VLLM_LOG="${VLLM_LOG:-logs/vllm.log}"
export VLLM_LOG
VLLM_PID="$(bash scripts/serve/serve_vllm.sh)"
[ -n "${VLLM_PID}" ] || { echo "ERROR: serve_vllm.sh produced no PID" >&2; exit 1; }
echo "[serve] vLLM running (pid=${VLLM_PID}); log: ${VLLM_LOG}"

echo "[serve] waiting for /health on :${VLLM_PORT} (up to $((SERVE_READY_TRIES * 5))s) ..."
READY=0
for _ in $(seq 1 "${SERVE_READY_TRIES}"); do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: serve process exited during startup" >&2; exit 1
    fi
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
        READY=1; break
    fi
    sleep 5
done
[ "${READY}" = "1" ] || { echo "ERROR: server not ready after $((SERVE_READY_TRIES * 5))s" >&2; exit 1; }
echo "[serve] ready."

echo "[eval] launching run.sh (tasks=${TASKS} n=${N_CONCURRENT})"
export MODEL="litellm_proxy/${MODEL}"
export MODEL_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export MODEL_API_KEY="EMPTY"
export N_CONCURRENT="${N_CONCURRENT}"
export TASK_PATH="${TASKS}"
export QUIET="${QUIET:-1}"
[ -n "${N_TASKS:-}" ] && export N_TASKS
export AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-1.0}"
bash scripts/eval/run.sh
echo "[eval] run.sh finished (exit $?)"
