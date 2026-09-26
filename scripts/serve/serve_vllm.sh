#!/usr/bin/env bash
# Serve vLLM inside the CUDA SIF, on this node. Owns the full serve invocation:
# per-model serve-arg loading, node-local cache wiring, the apptainer exec, and
# backgrounding. Prints the vLLM PID to stdout (nothing else on stdout) so a caller
# (run.sbatch) can capture $VLLM_PID for readiness checks and cleanup. All logs go
# to logs/vllm.log (and diagnostics to stderr).
#
# Usage (from the repo root):
#   VLLM_CUDA_SIF=/path/vllm.sif MODEL=Qwen/Qwen3.5-9B bash scripts/serve/serve_vllm.sh
#   VLLM_PID=$(bash scripts/serve/serve_vllm.sh)   # capture the PID
#
# Knobs (env vars):
#   VLLM_CUDA_SIF    CUDA vLLM Singularity image (required).
#   MODEL            HF repo to serve (default Qwen/Qwen3.5-9B).
#   VLLM_PORT        serve port (default 8000).
#   TENSOR_PARALLEL  TP (default 1); DATA_PARALLEL DP (default 1).
#   MAX_MODEL_LEN    context length (default 131072; overridden by serve config).
#   GPU_MEM_UTIL     gpu-memory-utilization (default 0.85).
#   BASE_DIR         node-local base (default ${PBS_LOCALDIR:-${TMPDIR:-/tmp}}).
#   CACHE_DIR        single host cache dir bound at /cache (default ${BASE_DIR}/.cache).
#   SERVE_CONFIG     per-model serve json (default configs/serve/<MODEL>.json).
#   VLLM_EXTRA_ARGS  explicit vLLM flags; if empty, auto-loaded from ${SERVE_CONFIG}.
#   VLLM_LOG         vLLM output file (default logs/vllm.log).
#   APPTAINER_BIN    override the container tool (auto: apptainer > singularity).
set -euo pipefail
cd "$(dirname "$0")/../.."

VLLM_CUDA_SIF="${VLLM_CUDA_SIF:-}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
VLLM_PORT="${VLLM_PORT:-8000}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
DATA_PARALLEL="${DATA_PARALLEL:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
BASE_DIR="${BASE_DIR:-${PBS_LOCALDIR:-${TMPDIR:-/tmp}}}"
CACHE_DIR="${CACHE_DIR:-${BASE_DIR}/.cache}"
SERVE_CONFIG="${SERVE_CONFIG:-configs/serve/${MODEL}.json}"
VLLM_LOG="${VLLM_LOG:-logs/vllm.log}"
APPTAINER_BIN="${APPTAINER_BIN:-$(command -v apptainer || command -v singularity || echo singularity)}"

[ -n "${VLLM_CUDA_SIF}" ] || { echo "ERROR: VLLM_CUDA_SIF not set" >&2; exit 2; }
[ -f "${VLLM_CUDA_SIF}" ] || { echo "ERROR: VLLM_CUDA_SIF not found: ${VLLM_CUDA_SIF}" >&2; exit 2; }
command -v "${APPTAINER_BIN}" >/dev/null 2>&1 || { echo "ERROR: container tool not found: ${APPTAINER_BIN}" >&2; exit 127; }

# --- per-model serve args (configs/serve/<MODEL>.json) --------------------------
# Model-intrinsic serving flags (tool/reasoning parsers, dtype, context length) travel
# WITH the model; node knobs (TP/DP/gpu-mem) stay in the env. Auto-select by MODEL so
# MODEL=IFM/K2-Horizon-32B picks IFM/K2-Horizon-32B.json. Empty VLLM_EXTRA_ARGS here
# means "load from the JSON"; a non-empty env VLLM_EXTRA_ARGS is explicit and wins.
if [ -z "${VLLM_EXTRA_ARGS:-}" ] && [ -f "${SERVE_CONFIG}" ]; then
    echo "[serve] loading vLLM serve args from ${SERVE_CONFIG}" >&2
    eval "$(python3 - "${SERVE_CONFIG}" <<'PY'
import json, os, shlex, sys
c = json.load(open(sys.argv[1]))
def emit(var, val):
    if os.environ.get(var):
        return
    print(f"{var}={shlex.quote(val)}")
if isinstance(c.get("extra_args"), list):
    emit("VLLM_EXTRA_ARGS", " ".join(str(a) for a in c["extra_args"]))
if c.get("max_model_len"):
    emit("MAX_MODEL_LEN", str(c["max_model_len"]))
PY
)"
elif [ -f "${SERVE_CONFIG}" ] && [ -n "${VLLM_EXTRA_ARGS:-}" ]; then
    echo "[serve] using explicit VLLM_EXTRA_ARGS (override); serve config ${SERVE_CONFIG} exists" >&2
elif [ ! -f "${SERVE_CONFIG}" ]; then
    echo "[serve] no serve config ${SERVE_CONFIG}; using defaults (qwen3_coder parser)" >&2
fi

# ABCI/PBS can hand GPUs as UUIDs; vLLM needs integer indices.
case "${CUDA_VISIBLE_DEVICES:-}" in
    *[!0-9,]*)
        _NGPU=$(printf '%s' "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
        export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((_NGPU - 1)))"
        echo "[serve] normalized CUDA_VISIBLE_DEVICES -> ${CUDA_VISIBLE_DEVICES}" >&2 ;;
esac

echo "[serve] launching vLLM ${MODEL} on 0.0.0.0:${VLLM_PORT} (TP=${TENSOR_PARALLEL} DP=${DATA_PARALLEL})" >&2
mkdir -p "${CACHE_DIR}" "$(dirname "${VLLM_LOG}")"
export APPTAINER_TMPDIR="${BASE_DIR}" SINGULARITY_TMPDIR="${BASE_DIR}"
# One host cache dir (node-local scratch, not quota-limited HOME) bound at /cache; all of
# HF models, vLLM's torch_compile_cache, and tmp live under it so nothing touches HOME.
"${APPTAINER_BIN}" exec --nv --writable-tmpfs \
    --workdir "${BASE_DIR}" \
    --bind "${CACHE_DIR}:/cache" \
    --env TMPDIR=/cache/tmp \
    --env HF_HOME=/cache/huggingface \
    --env VLLM_CACHE_ROOT=/cache/vllm \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
    "${VLLM_CUDA_SIF}" \
    vllm serve "${MODEL}" \
        --host 0.0.0.0 \
        --port "${VLLM_PORT}" \
        --tensor-parallel-size "${TENSOR_PARALLEL}" \
        --data-parallel-size "${DATA_PARALLEL}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        ${VLLM_EXTRA_ARGS:-} \
        >> "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

# PID handoff: print ONLY the vLLM pid on stdout so `VLLM_PID=$(...serve_vllm.sh)`
# captures it cleanly. Caller owns readiness checking and final cleanup.
echo "${VLLM_PID}"