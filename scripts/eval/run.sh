#!/usr/bin/env bash
# Run mini-swe-agent on a local LeetCode task directory under Harbor + Singularity,
# using config/leetcode.yaml for the agent prompts/model settings.
#
# Adapts the old one-shot (custom SimpleCodeAgent on benchmarks/leetcode/generated/cpp)
# to the built-in mini-swe-agent and a local `-p` task path (e.g. a generated Python
# dataset). mini-swe-agent consumes config/leetcode.yaml via the `config_file` agent
# kwarg; the model is reached through your OpenAI-compatible server (vLLM), so only
# OPENAI_BASE_URL/OPENAI_API_KEY are needed (no LANGUAGE/SOLUTION_FILE, which were
# specific to SimpleCodeAgent).
#
# Usage:
#   scripts/eval/run_leetcode.sh [-p TASK_PATH] [--dry-run]
#
#   -p TASK_PATH   local task or dataset directory (default: benchmarks/leetcode/
#                  datasets/python/shortest-distance-after-road-addition-queries-i)
#   --dry-run      print the resolved harbor JobConfig and exit (harbor --print-config)
#
# Knobs (env vars), mirroring scripts/eval/run.sh:
#   MODEL            litellm model id to serve (default openai/Qwen/Qwen3.5-9B)
#   MODEL_BASE_URL   OpenAI-compatible base URL (default http://127.0.0.1:46977/v1)
#   MODEL_API_KEY    key for that server (default dummy)
#   CONFIG_FILE      mini-swe-agent config yaml (default config/leetcode.yaml)
#   JOB_NAME         harbor job name (default leetcode-<lang>-<timestamp>)
#   JOBS_DIR         output directory (default jobs)
#   N_CONCURRENT     parallel trials (default 1)
#   AGENT_TIMEOUT_MULT  multiplier for task agent timeout (default 1.0; e.g. 12 -> 12x)
#   QUIET              set to 1 to suppress harbor's live progress renderer (default 0)
set -euo pipefail
cd "$(dirname "$0")/../.."
# Keep uv's package cache off NFS/home. Babel /scratch is node-local.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/$USER/uv-cache}"
mkdir -p "$UV_CACHE_DIR"

TASK_PATH="${TASK_PATH:-benchmarks/leetcode/tasks}"
MODEL="${MODEL:-openai/Qwen/Qwen3.5-9B}"
MODEL_BASE_URL="${MODEL_BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL_API_KEY="${MODEL_API_KEY:-dummy}"
CONFIG_FILE="${CONFIG_FILE:-config/leetcode.yaml}"
JOB_NAME="${JOB_NAME:-leetcode-$(date +%Y%m%d-%H%M%S)}"
JOBS_DIR="${JOBS_DIR:-jobs}"
N_CONCURRENT="${N_CONCURRENT:-1}"
DRY_RUN="${DRY_RUN:-0}"
AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-1.0}"
QUIET="${QUIET:-0}"
AGENT="${AGENT:-mini-swe-agent}"
MAX_TOKENS="${MAX_TOKENS:-8192}"

[ -d "${TASK_PATH}" ] || { echo "ERROR: task path not found: ${TASK_PATH}" >&2; exit 2; }
[ -f "${CONFIG_FILE}" ] || { echo "ERROR: config file not found: ${CONFIG_FILE}" >&2; exit 2; }

# Resolve harbor. Prefer the project venv (.venv/bin/harbor): it carries the
# harbor_singularity_hpc package (pinned in pyproject [tool.uv.sources]) needed by
# `-e harbor_singularity_hpc...`, which the global `uv tool` harbor on PATH does NOT
# have ("No module named 'harbor_singularity_hpc'"). Fall back to PATH/uv otherwise.
if   [ -n "${HARBOR_BIN:-}" ];              then HARBOR_CMD=( "${HARBOR_BIN}" )
elif [ -x ".venv/bin/harbor" ];             then HARBOR_CMD=( .venv/bin/harbor )
elif command -v harbor >/dev/null 2>&1;     then HARBOR_CMD=( harbor )
elif command -v uv >/dev/null 2>&1;         then HARBOR_CMD=( uv run harbor )
else echo "ERROR: harbor not found. Run: uv sync" >&2; exit 127; fi

ARGS=(
    run
    -p "${TASK_PATH}"
    -a "${AGENT}"
    -m "${MODEL}"
    -e "harbor_singularity_hpc.environment:SingularityWritableEnvironment"
    --ek "singularity_no_mount=home,tmp"
    --ak "config_file=${CONFIG_FILE}"
    --ae "OPENAI_BASE_URL=${MODEL_BASE_URL}"
    --ae "OPENAI_API_KEY=${MODEL_API_KEY}"
    --ae "MAX_TOKENS=${MAX_TOKENS}"
    -n "${N_CONCURRENT}"
    --job-name "${JOB_NAME}"
    -o "${JOBS_DIR}"
    --agent-timeout-multiplier "${AGENT_TIMEOUT_MULT}"
    -y
)
[ "${QUIET}" = "1" ] && ARGS+=( --quiet )
[ "${DRY_RUN}" = "1" ] && ARGS+=( --print-config )

echo "task:  ${TASK_PATH}"
echo "harbor: ${HARBOR_CMD[0]}"
echo "model: ${MODEL}  base_url: ${MODEL_BASE_URL}  config: ${CONFIG_FILE}"
echo "+ ${HARBOR_CMD[*]} ${ARGS[*]}"
exec "${HARBOR_CMD[@]}" "${ARGS[@]}"
