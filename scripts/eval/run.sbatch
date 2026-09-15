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
#   GPU_MEM_UTIL     gpu-memory-utilization (default 0.85).
#   HF_HOME_BIND     host dir bound as container HF cache (default ~/.cache/huggingface).
#   BASE_DIR         host base dir for container scratch/tmp (default ${PBS_LOCALDIR:-${TMPDIR:-/tmp}}).
#   N_CONCURRENT     passed to run_leetcode.sh (default 32).
#   TASKS            task/dataset path (default benchmarks/leetcode/tasks).
#   N_TASKS          subset limit for a smoke run; empty = all.
set -euo pipefail
# Resolve the repo root from THIS script's location (<repo>/scripts/eval/...) so the
# relative `scripts/eval/run_leetcode.sh` below always resolves, regardless of submit
# dir (PBS_O_WORKDIR may point outside the repo on an interactive node).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
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
HF_HOME_BIND="${HF_HOME_BIND:-${HOME}/.cache/huggingface}"
BASE_DIR="${BASE_DIR:-${PBS_LOCALDIR:-${TMPDIR:-/tmp}}}"
N_CONCURRENT="${N_CONCURRENT:-32}"
TASKS="${TASKS:-benchmarks/leetcode/tasks}"
# Default job name: <dataset>--<model>, with any / in the model replaced by --.
DATASET_NAME="$(basename "${TASKS}")"
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

# ABCI/PBS can hand GPUs as UUIDs; vLLM needs integer indices.
case "${CUDA_VISIBLE_DEVICES:-}" in
    *[!0-9,]*)
        _NGPU=$(printf '%s' "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
        export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((_NGPU - 1)))"
        echo "[serve] normalized CUDA_VISIBLE_DEVICES -> ${CUDA_VISIBLE_DEVICES}" ;;
esac

echo "[serve] launching vLLM ${MODEL} on 0.0.0.0:${VLLM_PORT} (TP=${TENSOR_PARALLEL} DP=${DATA_PARALLEL})"
mkdir -p "${HF_HOME_BIND}" "${BASE_DIR}/tmp"
export APPTAINER_TMPDIR="${BASE_DIR}" SINGULARITY_TMPDIR="${BASE_DIR}"
"${APPTAINER_BIN}" exec --nv --writable-tmpfs \
    --workdir "${BASE_DIR}" \
    --bind "${HF_HOME_BIND}:/root/.cache/huggingface" \
    --bind "${BASE_DIR}/tmp:/tmp_work" \
    --env TMPDIR=/tmp_work \
    --env HF_HOME=/root/.cache/huggingface \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
    "${VLLM_CUDA_SIF}" \
    vllm serve "${MODEL}" \
        --host 0.0.0.0 \
        --port "${VLLM_PORT}" \
        --tensor-parallel-size "${TENSOR_PARALLEL}" \
        --data-parallel-size "${DATA_PARALLEL}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        >> "logs/vllm.log" 2>&1 &
VLLM_PID=$!

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

echo "[eval] launching run_leetcode.sh (tasks=${TASKS} n=${N_CONCURRENT})"
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
