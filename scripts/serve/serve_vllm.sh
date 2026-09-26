#!/usr/bin/env bash
# Serve vLLM on this node's GPUs, running the binary inside the CUDA vLLM sif.
# Reused by scripts/eval/run.sbatch (which backgrounds this and waits on /health).
# Foreground-exec model: this script runs vllm in the FOREGROUND; the caller is
# responsible for backgrounding it (run.sbatch does `bash serve_vllm.sh &`) and for
# stopping it.
#
# Usage (from the repo root):
#   VLLM_CUDA_SIF=/path/vllm.sif MODEL=Qwen/Qwen3.5-9B PORT=8000 \
#       bash scripts/serve/serve_vllm.sh
#
# Knobs (env vars) -- node/accelerator knobs only:
#   VLLM_CUDA_SIF    CUDA vLLM Singularity image (default $PWD/vllm-openai-cuda-*.sif
#                    is NOT assumed; set it, or SIF_PATH for a named base).
#   SIF_PATH         alias for VLLM_CUDA_SIF (legacy).
#   MODEL            HF repo to serve (default Qwen/Qwen3.5-9B).
#   PORT             serve port (default 8000).
#   TENSOR_PARALLEL  TP (default 1); DATA_PARALLEL DP (default 8).
#   MAX_MODEL_LEN    context length (default 262144).
#   GPU_MEM_UTIL     gpu-memory-utilization (default 0.85).
#   BASE_DIR         node-local base for scratch (default ${PBS_LOCALDIR:-${TMPDIR:-/tmp}}).
#   CACHE_DIR        single host cache dir bound at /cache (default
#                    ${XDG_CACHE_HOME:-${BASE_DIR}}/.cache -- honors the .env XDG_CACHE_HOME).
#   VLLM_ARGS_EXTRA  model-intrinsic vLLM flags (parsers, dtype, context knobs). These
#                    come from configs/serve/<MODEL>.json via run.sbatch; set explicitly
#                    for manual/standalone runs.
#   APPTAINER_BIN    override the container tool (auto: apptainer > singularity).
set -euo pipefail
cd "$(dirname "$0")/../.."

VLLM_CUDA_SIF="${VLLM_CUDA_SIF:-${SIF_PATH:-}}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
DATA_PARALLEL="${DATA_PARALLEL:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
BASE_DIR="${BASE_DIR:-${PBS_LOCALDIR:-${TMPDIR:-/tmp}}}"
# Honor the .env XDG_CACHE_HOME (node-local scratch) as the cache root; fall back to
# BASE_DIR (or /tmp) when unset (e.g. a login shell). Keeps caches off quota-limited HOME.
CACHE_DIR="${CACHE_DIR:-${XDG_CACHE_HOME:-${BASE_DIR}}/.cache}"
APPTAINER_BIN="${APPTAINER_BIN:-$(command -v apptainer || command -v singularity || echo singularity)}"

if [ -z "${VLLM_CUDA_SIF}" ]; then
    echo "ERROR: VLLM_CUDA_SIF (or SIF_PATH) not set." >&2
    echo "       Build one first: scripts/build/build_vllm.sh cuda" >&2
    exit 2
fi
[ -f "${VLLM_CUDA_SIF}" ] || { echo "ERROR: sif not found: ${VLLM_CUDA_SIF}" >&2; exit 2; }
command -v "${APPTAINER_BIN}" >/dev/null 2>&1 || { echo "ERROR: container tool not found: ${APPTAINER_BIN}" >&2; exit 127; }

# ABCI/PBS can hand GPUs as UUIDs; vLLM needs integer indices.
case "${CUDA_VISIBLE_DEVICES:-}" in
    *[!0-9,]*)
        _NGPU=$(printf '%s' "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
        export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((_NGPU - 1)))"
        echo "[serve] normalized CUDA_VISIBLE_DEVICES -> ${CUDA_VISIBLE_DEVICES}" >&2 ;;
esac

echo "[serve] launching vLLM ${MODEL} :${PORT} (TP=${TENSOR_PARALLEL} DP=${DATA_PARALLEL}) in ${VLLM_CUDA_SIF}" >&2
mkdir -p "${CACHE_DIR}"
export APPTAINER_TMPDIR="${BASE_DIR}" SINGULARITY_TMPDIR="${BASE_DIR}"
# One host cache dir (node-local scratch, not quota-limited HOME) bound at /cache; all
# of HF models, vLLM's torch_compile_cache, and tmp live under it so nothing touches HOME.
exec "${APPTAINER_BIN}" exec --nv --writable-tmpfs \
    --workdir "${BASE_DIR}" \
    --bind "${CACHE_DIR}:/cache" \
    --env TMPDIR=/cache/tmp \
    --env HF_HOME=/cache/huggingface \
    --env VLLM_CACHE_ROOT=/cache/vllm \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
    "${VLLM_CUDA_SIF}" \
    vllm serve "${MODEL}" \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --tensor-parallel-size "${TENSOR_PARALLEL}" \
        --data-parallel-size "${DATA_PARALLEL}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        ${VLLM_ARGS_EXTRA:-}