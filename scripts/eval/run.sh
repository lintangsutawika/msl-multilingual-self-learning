#!/usr/bin/env bash
# Minimal harbor launcher: mini-swe-agent on SWE-bench Multilingual (300 tasks).
#
# Usage:
#   MODEL=openai/Qwen/Qwen3-Coder-30B-A3B-Instruct \
#   MODEL_BASE_URL=http://localhost:8000/v1 MODEL_API_KEY=dummy \
#   scripts/eval/run.sh
#
# The dataset id `swebench_multilingual` resolves against harbor's default registry
# (hub.harborframework.com), 300 tasks, v1.0. Tasks are Dockerfile-defined
# (FROM swebench/sweb.eval.x86_64.<instance>), so ENV=singularity relies on the
# harbor-singularity-hpc environment (FROM fallback + writable rootfs); ENV=docker/modal
# build the Dockerfile natively.
set -euo pipefail
cd "$(dirname "$0")/../.."

# --- knobs ----------------------------------------------------------------------
DATASET="${DATASET:-swebench_multilingual}"
MODEL="${MODEL:-}"                       # REQUIRED, e.g. openai/<served-name> or anthropic/claude-...
MODEL_BASE_URL="${MODEL_BASE_URL:-}"     # set for an OpenAI-compatible server (vLLM); empty => native provider
MODEL_API_KEY="${MODEL_API_KEY:-dummy}"  # key for that server / provider
ENV="${ENV:-singularity}"                # singularity | docker | modal | ...
N_CONCURRENT="${N_CONCURRENT:-4}"        # parallel trials
N_TASKS="${N_TASKS:-}"                   # empty => all 300; set a small number to smoke-test
JOB_NAME="${JOB_NAME:-msl-$(date +%Y%m%d-%H%M%S)}"
JOBS_DIR="${JOBS_DIR:-jobs}"

# Singularity-only. Keep bind-paths mounted (so /etc/resolv.conf reaches the container
# and in-container pip has DNS); only suppress home,tmp. Leave the cache empty for
# node-local ($PBS_LOCALDIR/$SLURM_TMPDIR, resume-safe) or set a shared-FS path to persist.
SINGULARITY_NO_MOUNT="${SINGULARITY_NO_MOUNT:-home,tmp}"
SINGULARITY_CACHE_DIR="${SINGULARITY_CACHE_DIR:-}"
SINGULARITY_ENV_IMPORT_PATH="${SINGULARITY_ENV_IMPORT_PATH:-harbor_singularity_hpc.environment:SingularityWritableEnvironment}"

# mini-swe-agent needs Python >=3.11 (imports typing.NotRequired); many SWE-bench task
# images ship 3.10/3.9, and harbor installs the agent with an unpinned `uv tool install`
# against the image's system Python -> ImportError. Force a 3.11 interpreter for the agent
# tool venv. Set MSWEA_UV_PYTHON="" to disable.
MSWEA_UV_PYTHON="${MSWEA_UV_PYTHON:-3.11}"

[ -n "${MODEL}" ] || { echo "ERROR: set MODEL (e.g. MODEL=openai/<name> with MODEL_BASE_URL, or MODEL=anthropic/claude-...)." >&2; exit 2; }

# --- resolve harbor -------------------------------------------------------------
if   [ -n "${HARBOR_BIN:-}" ];              then HARBOR_CMD=( "${HARBOR_BIN}" )
elif command -v harbor >/dev/null 2>&1;     then HARBOR_CMD=( harbor )
elif [ -x ".venv/bin/harbor" ];             then HARBOR_CMD=( .venv/bin/harbor )
elif command -v uv >/dev/null 2>&1;         then HARBOR_CMD=( uv run harbor )
else echo "ERROR: harbor not found. Run: uv sync" >&2; exit 127; fi

# --- assemble command -----------------------------------------------------------
ARGS=(
    run
    -d "${DATASET}"
    -a mini-swe-agent
    -m "${MODEL}"
    -n "${N_CONCURRENT}"
    --job-name "${JOB_NAME}"
    -o "${JOBS_DIR}"
    -y
)

# Point the agent's litellm at the model endpoint (only when a base URL is given; for a
# native provider like anthropic/, leave it and set that provider's key in your env).
if [ -n "${MODEL_BASE_URL}" ]; then
    ARGS+=(
        --ae "OPENAI_BASE_URL=${MODEL_BASE_URL}"
        --ae "OPENAI_API_KEY=${MODEL_API_KEY}"
        --ae "MSWEA_API_KEY=${MODEL_API_KEY}"
    )
fi
[ -n "${MSWEA_UV_PYTHON}" ] && ARGS+=( --ae "UV_PYTHON=${MSWEA_UV_PYTHON}" )
[ -n "${N_TASKS}" ] && ARGS+=( -l "${N_TASKS}" )

# Environment: singularity is our custom writable-rootfs class (harbor's -e enum can't
# name a custom class), selected by import path; everything else uses -e.
case "${ENV}" in
    singularity*)
        ARGS+=( --environment-import-path "${SINGULARITY_ENV_IMPORT_PATH}" )
        ARGS+=( --ek "singularity_no_mount=${SINGULARITY_NO_MOUNT}" )
        [ -n "${SINGULARITY_CACHE_DIR}" ] && ARGS+=( --ek "singularity_image_cache_dir=${SINGULARITY_CACHE_DIR}" )
        ;;
    *)
        ARGS+=( -e "${ENV}" )
        ;;
esac

echo "DATASET=${DATASET}  MODEL=${MODEL}  ENV=${ENV}  n=${N_CONCURRENT}  tasks=${N_TASKS:-all}"
echo "+ ${HARBOR_CMD[*]} ${ARGS[*]}"
exec "${HARBOR_CMD[@]}" "${ARGS[@]}"
