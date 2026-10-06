#!/usr/bin/env bash
# One scheduler job that serves vLLM on this node's GPUs (inside the repo's vllm sif)
# and runs the mini-swe-agent Harbor eval against it, on the SAME node. Delegates
# serving to scripts/serve/serve_vllm.sh and evaluation to scripts/eval/run.sh.
# The eval runs with -e SingularityWritableEnvironment, so the agent's task container
# runs ON this node and reaches the served vLLM at localhost directly.
#
# Two modes:
#   * DELIBERATOR_BASE_URL set  -> use that existing OpenAI-compatible endpoint; no
#     local server, no GPUs.
#   * DELIBERATOR_BASE_URL unset -> spawn vLLM here (build/serve), eval against it.
#
# PBS (ABCI) + SLURM supported. ABCI: qsub (PBS takes no positional args; pass env
# via -v). SLURM: sbatch (flip the #SBATCH header block / pass CLI flags).
#
# Usage (ABCI qsub):
#   qsub -j oe -o logs/ -P gai51740 -q R9920261300 -v RTYPE=rt_HF \
#       -l select=1 -l walltime=48:00:00 -V scripts/eval/run.sbatch
#   Set env via -v or in .env: VLLM_CUDA_SIF, MODEL, TENSOR_PARALLEL, DATA_PARALLEL,
#   MAX_MODEL_LEN, N_CONCURRENT, TASKS, RESUME_FILTER_ERRORS, AGENT_TIMEOUT_MULT.
#
# Knobs (env vars):
#   VLLM_CUDA_SIF    CUDA vLLM Singularity image (required to serve locally).
#   MODEL            bare HF repo to serve + eval, e.g. Qwen/Qwen3.5-9B.
#   VLLM_PORT        serve port (default: a randomly-chosen free port).
#   TENSOR_PARALLEL / DATA_PARALLEL / MAX_MODEL_LEN / GPU_MEM_UTIL  serve knobs
#   TASKS            task/dataset path for run.sh (default benchmarks/leetcode/tasks-b).
#   N_TASKS          subset limit for a smoke run; empty = all.
#   N_CONCURRENT     passed to run.sh (default 16).
#   QUIET            suppress harbor live renderer (default 1 under a batch).
#   AGENT_TIMEOUT_MULT  agent timeout multiplier (default 6: 6 x 300s task timeout = 1800s trial budget).
#   RESUME_FILTER_ERRORS  space-separated error types to retry on resume.
#   RESUME_PASSES    resume the job this many more times after the first pass, while
#                    vLLM is still up (default 0: one pass, as before).
#   DELIBERATOR_BASE_URL  if set, skip local serve and eval against this endpoint.
#   SERVE_CONFIG     per-model serve json (default configs/serve/<MODEL>.json).
#   SERVE_READY_TRIES  /health poll budget (tries x 5s; default 180 ~ 15 min).
set -euo pipefail

# A batch script's $0 is the spooled copy; resolve the repo from the submit dir.
# PBS/Torque (ABCI) sets PBS_O_WORKDIR; SLURM sets SLURM_SUBMIT_DIR.
REPO_ROOT="${REPO_ROOT:-}"
if [ -z "${REPO_ROOT}" ]; then
    if [ -n "${PBS_O_WORKDIR:-}" ] && [ -x "${PBS_O_WORKDIR}/scripts/eval/run.sh" ]; then
        REPO_ROOT="${PBS_O_WORKDIR}"
    elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -x "${SLURM_SUBMIT_DIR}/scripts/eval/run.sh" ]; then
        REPO_ROOT="${SLURM_SUBMIT_DIR}"
    else
        REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    fi
fi
cd "${REPO_ROOT}"

# Live log under PBS: PBS spools stdout and only copies back at job end, so tee to a
# shared-FS file you can tail -f while it runs.
if [ -n "${PBS_JOBID:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p logs
    exec > >(tee -a "logs/${PBS_JOBID}.out") 2>&1
    echo "[log] teeing output to logs/${PBS_JOBID}.out  (tail -f it)"
fi
[ -f .env ] && . .env

# --- deliberator / serving decision ---------------------------------------------
DELIBERATOR_BASE_URL="${DELIBERATOR_BASE_URL:-}"
if [ -n "${DELIBERATOR_BASE_URL}" ]; then SERVE_DEFAULT=0; else SERVE_DEFAULT=1; fi
SERVE_VLLM="${SERVE_VLLM:-${SERVE_DEFAULT}}"

# MODEL is the bare HF repo. Do NOT put openai/ in it (serve_vllm.sh serves the bare id).
MODEL="${MODEL:-${VLLM_MODEL:-Qwen/Qwen3.5-9B}}"
export MODEL

# --- per-model vLLM serve params (configs/serve/<MODEL>.json) --------------------
# Model-INTINSIC serving flags (tool/reasoning parsers, dtype, context length) travel
# WITH the model; node knobs (TP/DP/gpu-mem/sif) stay in the env. Auto-loaded unless
# an explicit VLLM_ARGS_EXTRA is already set (env override wins).
VLLM_ARGS_EXTRA="${VLLM_ARGS_EXTRA:-}"
SERVE_CONFIG="${SERVE_CONFIG:-configs/serve/${MODEL}.json}"
if [ -z "${VLLM_ARGS_EXTRA:-}" ] && [ -f "${SERVE_CONFIG}" ]; then
    echo "[serve] loading vLLM serve params from ${SERVE_CONFIG}"
    eval "$(python3 - "${SERVE_CONFIG}" <<'PY'
import json, os, shlex, sys
c = json.load(open(sys.argv[1]))
def emit(var, val):
    if os.environ.get(var):
        return
    print(f"{var}={shlex.quote(val)}")
if isinstance(c.get("extra_args"), list):
    emit("VLLM_ARGS_EXTRA", " ".join(str(a) for a in c["extra_args"]))
if c.get("max_model_len"):
    emit("MAX_MODEL_LEN", str(c["max_model_len"]))
PY
)"
elif [ -z "${VLLM_ARGS_EXTRA:-}" ]; then
    echo "[serve] no serve config ${SERVE_CONFIG}; defaulting to qwen3_coder parser" >&2
fi

export QUIET="${QUIET:-1}"

# Pick an ephemeral free TCP port so concurrent jobs on a shared node don't collide.
pick_free_port() {
    local p
    for _ in $(seq 1 64); do
        p=$(( (RANDOM % 16384) + 49152 ))
        ss -Htln 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${p}\$" && continue
        echo "${p}"; return 0
    done
    echo "${p}"
}
SERVE_PORT="${VLLM_PORT:-$(pick_free_port)}"

VLLM_PID=""
EVAL_PID=""
_CLEANED_UP=0
cleanup() {
    [ "${_CLEANED_UP}" = "1" ] && return; _CLEANED_UP=1
    # Stop eval FIRST (SIGINT) so in-flight trials write CancelledError result.jsons
    # (resumed next chunk) BEFORE pulling vLLM out from under them.
    if [ -n "${EVAL_PID}" ] && kill -0 "${EVAL_PID}" 2>/dev/null; then
        echo "[cleanup] stopping eval (pid=${EVAL_PID}); cancelling in-flight trials"
        kill -INT "${EVAL_PID}" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "${EVAL_PID}" 2>/dev/null || break; sleep 1; done
        kill -TERM "${EVAL_PID}" 2>/dev/null || true
    fi
    [ -n "${VLLM_PID}" ] && { echo "[cleanup] stopping vLLM (pid=${VLLM_PID})";
        kill "${VLLM_PID}" 2>/dev/null; wait "${VLLM_PID}" 2>/dev/null; }
}
trap cleanup EXIT SIGTERM SIGINT

if [ "${SERVE_VLLM}" = "1" ]; then
    echo "[serve] launching vLLM ${MODEL} on 127.0.0.1:${SERVE_PORT} via scripts/serve/serve_vllm.sh"
    VLLM_CUDA_SIF="${VLLM_CUDA_SIF:-}" MODEL="${MODEL}" PORT="${SERVE_PORT}" \
        BASE_DIR="${BASE_DIR:-${PBS_LOCALDIR:-${TMPDIR:-/tmp}}}" \
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}" \
        VLLM_ARGS_EXTRA="${VLLM_ARGS_EXTRA:-}" \
        bash scripts/serve/serve_vllm.sh > /dev/null 2>&1 &
    VLLM_PID=$!

    SERVE_READY_TRIES="${SERVE_READY_TRIES:-180}"   # ~15 min at 5s
    echo "[serve] waiting for /health on :${SERVE_PORT} (up to $((SERVE_READY_TRIES * 5))s) ..."
    READY=0
    for _ in $(seq 1 "${SERVE_READY_TRIES}"); do
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "ERROR: serve process exited during startup" >&2; exit 1
        fi
        if curl -sf "http://127.0.0.1:${SERVE_PORT}/health" >/dev/null 2>&1; then
            READY=1; break
        fi
        sleep 5
    done
    [ "${READY}" = "1" ] || { echo "ERROR: server not ready after $((SERVE_READY_TRIES * 5))s" >&2; exit 1; }
    echo "[serve] ready."

    export DELIBERATOR_BASE_URL="http://127.0.0.1:${SERVE_PORT}/v1"
else
    echo "[serve] using provided endpoint ${DELIBERATOR_BASE_URL} (no local vLLM)"
    export DELIBERATOR_BASE_URL
fi

echo "================================================================"
echo "Harbor eval: model=${MODEL}  endpoint=${DELIBERATOR_BASE_URL}"
echo "================================================================"

# Delegate to the eval runner. Run in the BACKGROUND and wait, so a scheduler TERM
# reaches our trap while run.sh (exec'd into harbor) is still a killable child.
# RESUME_PASSES (default 0) then resumes the job that many more times while vLLM is
# still up, so trials that ended in a RESUME_FILTER_ERRORS error are re-run.
run_eval() {
    TASK_PATH="${TASKS:-benchmarks/leetcode/tasks-b}" \
    MODEL_BASE_URL="${DELIBERATOR_BASE_URL}" MODEL_API_KEY="EMPTY" \
    MODEL="litellm_proxy/${MODEL}" \
    N_CONCURRENT="${N_CONCURRENT:-16}" \
    AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-6.0}" \
    RESUME_FILTER_ERRORS="${RESUME_FILTER_ERRORS:-}" \
    QUIET="${QUIET:-1}" \
    "$@" bash scripts/eval/run.sh &
    EVAL_PID=$!
    local status=0
    wait "${EVAL_PID}" || status=$?
    echo "[eval] run.sh finished (exit ${status})"
}
run_eval
for _pass in $(seq 1 "${RESUME_PASSES:-0}"); do
    echo "[eval] resume pass ${_pass}/${RESUME_PASSES}"
    run_eval env RESUME=1
done