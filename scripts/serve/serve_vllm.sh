#!/usr/bin/env bash
# Serve a model with vLLM, running the binary inside the base sif built by
# scripts/build/vllm.sh (apptainer exec --nv). The sif carries its own vllm + CUDA
# runtime, so this does not need a vllm install on the host.
#
# Usage:
#   scripts/serve_vllm.sh                      # defaults below
#   MODEL=Qwen/Qwen3.5-9B PORT=8000 scripts/serve_vllm.sh
#
# Knobs:
#   SIF_PATH   base vllm sif to exec into (default scripts/build/vllm.sif)
#   MODEL      HF model id / local path to serve
#   PORT       serve port (default 8000)
#   HF_CACHE   host dir exposed as the container HF cache (default ~/.cache/huggingface)
#   VLLM_ARGS_EXTRA  extra args appended to the default tensor-parallel/reasoning set
set -euo pipefail
cd "$(dirname "$0")/.."

# Resolve the base sif: explicit SIF_PATH, else the build script's default output.
SIF_PATH="${SIF_PATH:-scripts/build/vllm.sif}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
PORT="${PORT:-8000}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"

if [ ! -f "${SIF_PATH}" ]; then
    echo "ERROR: base sif not found at ${SIF_PATH}. Build it first with scripts/build/vllm.sh." >&2
    exit 2
fi

# Resolve the container runtime (must match what built the sif).
if command -v apptainer >/dev/null 2>&1;     then RUNTIME=( apptainer )
elif command -v singularity >/dev/null 2>&1; then RUNTIME=( singularity )
else echo "ERROR: neither apptainer nor singularity found on PATH." >&2; exit 127; fi

# Default serve args mirror the original scaffold (data-parallel 8 on this node; the
# 8x RTX A6000 tensor-parallel layout). Override the size with TENSOR_PARALLEL if needed.
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"

VLLM_ARGS=(
    serve "${MODEL}"
    --port "${PORT}"
    --data-parallel-size "${DATA_PARALLEL_SIZE}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
)
# shellcheck disable=SC2206
VLLM_ARGS+=( ${VLLM_ARGS_EXTRA:-} )

echo "sif:   ${SIF_PATH}"
echo "model: ${MODEL}  port: ${PORT}  hf_cache: ${HF_CACHE}"
echo "+ ${RUNTIME[*]} exec --nv --no-home --env PYTHONNOUSERSITE=1 --bind ${HF_CACHE}:/hf-cache --env HF_HOME=/hf-cache ${SIF_PATH} vllm ${VLLM_ARGS[*]}"
exec "${RUNTIME[@]}" exec --nv --no-home \
    --env PYTHONNOUSERSITE=1 \
    --bind "${HF_CACHE}:/hf-cache" \
    --env HF_HOME=/hf-cache \
    "${SIF_PATH}" vllm "${VLLM_ARGS[@]}"