#!/usr/bin/env bash
# Use a fresh task output directory/job name for each generated benchmark version.
set -euo pipefail
cd "$(dirname "$0")/../.."
TASKS="${TASKS:-benchmarks/leetcode/qwen-tasks-v2}"
AGENT="${AGENT:-model}"
JOB_NAME="${JOB_NAME:-leetcode-unified-$(date +%Y%m%d-%H%M%S)}"
args=(run -p "$TASKS" -n "${N_CONCURRENT:-1}" --job-name "$JOB_NAME" -o jobs -y)
if [[ "$AGENT" == oracle ]]; then
    args+=(-a oracle)
else
    args+=(-a msl_multilingual_self_learning.agents.simple_code_agent:SimpleCodeAgent
        -m "openai/${MODEL:-Qwen/Qwen3.5-9B}"
        --ae "OPENAI_BASE_URL=${MODEL_BASE_URL:-http://127.0.0.1:8000/v1}"
        --ae "OPENAI_API_KEY=${MODEL_API_KEY:-EMPTY}")
    if [[ -n "${LANGUAGE:-}" ]]; then
        args+=(--ae "LANGUAGE=$LANGUAGE")
    fi
fi
if [[ "${ENV:-singularity}" == singularity ]]; then
    args+=(--environment-import-path harbor_singularity_hpc.environment:SingularityWritableEnvironment
        --ek "singularity_no_mount=home,tmp")
else
    args+=(-e "$ENV")
fi
exec .venv/bin/harbor "${args[@]}"
