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
#   scripts/eval/run.sh [-p TASK_PATH] [--dry-run]
#
# Knobs (env vars):
#   MODEL            litellm model id to serve (default openai/Qwen/Qwen3.5-9B)
#   MODEL_BASE_URL   OpenAI-compatible base URL (default http://127.0.0.1:8000/v1)
#   MODEL_API_KEY    key for that server (default dummy)
#   CONFIG_FILE      mini-swe-agent config yaml (default config/leetcode.yaml; a
#                     per-model configs/sampling/<repo>.yaml is auto-selected by
#                     MODEL when present, e.g. MODEL=Qwen/Qwen3.5-9B ->
#                     configs/sampling/Qwen/Qwen3.5-9B.yaml).
#   JOB_NAME         harbor job name (default <dataset>_<model>-run-<RUN>; deterministic
#                     so resume works across chunks AND RUN=0,1,... gives repeat runs).
#   JOBS_DIR         output directory (default jobs)
#   N_CONCURRENT     parallel trials (default 1)
#   AGENT_TIMEOUT_MULT  multiplier for task agent timeout (default 1.0; e.g. 12 -> 12x)
#   MEMORY_MB           per-task container memory cap in MB, passed as
#                       --ek override_memory_mb (raises the watch/kill limit for
#                       big models; default = task.toml memory_mb, e.g. 16384).
#   MEMORY_ENFORCEMENT  memory watchdog policy (--ek memory_enforcement_policy).
#   QUIET              set to 1 to suppress harbor's live progress renderer (default 0)
#
# Resume (from the tts-tokens-that-suffice harness):
#   RESUME=auto|1|0    auto (default) resumes iff <JOBS_DIR>/<JOB_NAME>/config.json
#                      exists; 1 forces resume (errors if no config); 0 forces fresh.
#   RESUME_FILTER_ERRORS  space-separated error types to retry on resume, e.g.
#                      "CancelledError AgentTimeoutError NetworkConnectionError
#                       ApiOverloadedError AgentSetupTimeoutError RewardFileNotFoundError".
#                      Passed as `-f <et>` to `harbor job resume`. Setting this REPLACES
#                      harbor's default (CancelledError), so include CancelledError
#                      explicitly if you set it.
set -euo pipefail
cd "$(dirname "$0")/../.."
# Keep uv's package cache off NFS/home. Prefer XDG_CACHE_HOME (set in .env to the
# node-local PBS_LOCALDIR); fall back to /tmp when unset (e.g. a login shell). Never
# use /scratch -- it does not exist on ABCI.
CACHE_ROOT="${XDG_CACHE_HOME:-/tmp}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CACHE_ROOT}/uv-cache}"
mkdir -p "$UV_CACHE_DIR"

TASK_PATH="${TASK_PATH:-benchmarks/leetcode/tasks}"
MODEL="${MODEL:-openai/Qwen/Qwen3.5-9B}"
MODEL_BASE_URL="${MODEL_BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL_API_KEY="${MODEL_API_KEY:-dummy}"
# Bare HF repo (strip a litellm provider prefix like openai/ or litellm_proxy/),
# so per-model sampling configs resolve by the real repo path.
_MODEL_BARE="${MODEL#openai/}"
_MODEL_BARE="${_MODEL_BARE#litellm_proxy/}"
# Per-model sampling config: configs/sampling/<repo>.yaml wins when present;
# else fall back to config/leetcode.yaml. An explicit CONFIG_FILE always wins.
if [ -z "${CONFIG_FILE:-}" ] && [ -f "configs/sampling/${_MODEL_BARE}.yaml" ]; then
    CONFIG_FILE="configs/sampling/${_MODEL_BARE}.yaml"
fi
CONFIG_FILE="${CONFIG_FILE:-configs/task/leetcode.yaml}"
# Deterministic JOB_NAME so a resume works across walltime chunks AND RUN=0,1,2,...
# gives repeat (non-colliding) runs of the same data+model. Derived from TASK_PATH
# (dataset slug = its dir name) + MODEL (bare, / -> --). Set JOB_NAME to override.
RUN="${RUN:-0}"
_DATASET_SLUG="$(basename "${TASK_PATH%/}")"
JOB_NAME="${JOB_NAME:-${_DATASET_SLUG}_${_MODEL_BARE//\//--}-run-${RUN}}"
JOBS_DIR="${JOBS_DIR:-jobs}"
N_CONCURRENT="${N_CONCURRENT:-1}"
DRY_RUN="${DRY_RUN:-0}"
AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-1.0}"
QUIET="${QUIET:-0}"
AGENT="${AGENT:-mini-swe-agent}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-10200}"
MEMORY_MB="${MEMORY_MB:-}"
MEMORY_ENFORCEMENT="${MEMORY_ENFORCEMENT:-}"
RESUME="${RESUME:-auto}"
RESUME_FILTER_ERRORS="${RESUME_FILTER_ERRORS:-}"

JOB_DIR="${JOBS_DIR}/${JOB_NAME}"

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

# --- Resume vs fresh run --------------------------------------------------------
# If this job dir already has config.json, the job was started before (e.g. a previous
# allocation chunk) -> RESUME it with `harbor job resume` instead of `harbor run`.
# Resume reads the stored config (dataset/model/agent env), so the SAME command works
# every chunk. `-f <error-type>` drops trials whose last error is that type so they are
# retried rather than kept as failures. Default filter is CancelledError (what an
# interrupted trial becomes when the batch kills harbor at the walltime boundary).
if [ "${RESUME}" = "1" ] || { [ "${RESUME}" = "auto" ] && [ -f "${JOB_DIR}/config.json" ]; }; then
    RESUME_ARGS=( job resume -p "${JOB_DIR}" )
    for _et in ${RESUME_FILTER_ERRORS}; do RESUME_ARGS+=( -f "${_et}" ); done
    echo "RESUME  job=${JOB_NAME}  dir=${JOB_DIR}"
    echo "harbor: ${HARBOR_CMD[0]}"
    echo "retry-error-types=${RESUME_FILTER_ERRORS:-<harbor default: CancelledError>}"
    echo "+ ${HARBOR_CMD[*]} ${RESUME_ARGS[*]}"
    exec "${HARBOR_CMD[@]}" "${RESUME_ARGS[@]}"
fi

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
    --ae "REQUEST_TIMEOUT=${REQUEST_TIMEOUT}"
    -n "${N_CONCURRENT}"
    --job-name "${JOB_NAME}"
    -o "${JOBS_DIR}"
    --agent-timeout-multiplier "${AGENT_TIMEOUT_MULT}"
    -y
)
[ -n "${MEMORY_MB}" ] && ARGS+=( --ek "override_memory_mb=${MEMORY_MB}" )
[ -n "${MEMORY_ENFORCEMENT}" ] && ARGS+=( --ek "memory_enforcement_policy=${MEMORY_ENFORCEMENT}" )
[ "${QUIET}" = "1" ] && ARGS+=( --quiet )
[ "${DRY_RUN}" = "1" ] && ARGS+=( --print-config )

echo "task:  ${TASK_PATH}"
echo "harbor: ${HARBOR_CMD[0]}"
echo "model: ${MODEL}  base_url: ${MODEL_BASE_URL}  config: ${CONFIG_FILE}"
echo "+ ${HARBOR_CMD[*]} ${ARGS[*]}"
exec "${HARBOR_CMD[@]}" "${ARGS[@]}"
