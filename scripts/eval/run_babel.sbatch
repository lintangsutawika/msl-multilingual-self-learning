#!/usr/bin/env bash
# scripts/eval/run.sbatch on CMU Babel (SLURM): one 48 GB GPU, preemptible, requeued.
# Sets only what differs from ABCI, then runs scripts/eval/run.sbatch unchanged
# (vLLM from VLLM_CUDA_SIF with configs/serve/<MODEL>.json, then scripts/eval/run.sh).
#
# Submit from the repo root (mkdir -p logs first):
#   TASKS=benchmarks/leetcode/tasks-test-v3 MODEL=Qwen/Qwen3.5-9B \
#   AGENT=msl_multilingual_self_learning.agents.simple_code_agent:SimpleCodeAgent \
#       sbatch --export=ALL scripts/eval/run_babel.sbatch
#
# Site paths, e.g. in .env (git-ignored; also read by run.sbatch):
#   VENV_ARCHIVE   the repo's .venv packed as .tar.zst (tar -C <repo> -cf - .venv | zstd);
#                  unpacked to node-local disk, since importing from /data is very slow.
#   VENV_FROM      the checkout that archive was packed from (its #! lines are rewritten).
#   VLLM_CUDA_SIF  vLLM Singularity image.
#   CACHE_DIR      host dir bound at /cache by serve_vllm.sh; its huggingface/ holds the weights.
#
# Knobs: TASKS, MODEL (bare repo), AGENT, RUN, N_CONCURRENT (default 16), MEMORY_MB
# (task container memory, default 2048), RESUME_PASSES (default 2), RESUME_FILTER_ERRORS.
#
#SBATCH --job-name=eval
#SBATCH --account=teruko
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --constraint=VRAM_48GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

export REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from the repo root}"
cd "${REPO_ROOT}"
[ -f .env ] && . .env
: "${TASKS:?set TASKS, e.g. benchmarks/leetcode/tasks-test-v3}"
: "${VENV_ARCHIVE:?set VENV_ARCHIVE (see the header)}"
: "${VLLM_CUDA_SIF:?set VLLM_CUDA_SIF (see the header)}"
export TASKS VLLM_CUDA_SIF
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} restart=${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Node-local scratch for the venv and vLLM's temporary files; removed at exit.
SCRATCH="/scratch/${USER}/eval-${SLURM_JOB_ID}"
mkdir -p "${SCRATCH}"
trap 'rm -rf "${SCRATCH}"' EXIT
mkdir -p "${SCRATCH}/venv"
zstd -dc -T0 "${VENV_ARCHIVE}" | tar -C "${SCRATCH}/venv" -xf -
# Console scripts carry the packing checkout's interpreter path in their #! line.
VENV_FROM="${VENV_FROM:-${REPO_ROOT}}"
sed -i "1s|^#!${VENV_FROM}/.venv/bin/python|#!${SCRATCH}/venv/.venv/bin/python|" "${SCRATCH}"/venv/.venv/bin/* 2>/dev/null || true
export HARBOR_BIN="${SCRATCH}/venv/.venv/bin/harbor"
export PATH="${SCRATCH}/venv/.venv/bin:${PATH}"
# Run this checkout's code even if the venv's editable install points elsewhere.
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
echo "venv staged $(date)"

# One GPU: one vLLM replica (ABCI's default is 8).
export DATA_PARALLEL="${DATA_PARALLEL:-1}" TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
export BASE_DIR="${SCRATCH}"
export CACHE_DIR="${CACHE_DIR:-${HOME}/.cache}"
# A fixed port per job (tasks, model, agent, run): Harbor stores the model URL in the
# job, so a requeued or resubmitted job must find vLLM on the same port.
_KEY="${TASKS}|${MODEL:-}|${AGENT:-mini-swe-agent}|${RUN:-0}"
export VLLM_PORT="${VLLM_PORT:-$((20000 + $(printf '%s' "${_KEY}" | cksum | cut -d' ' -f1) % 20000))}"
export N_CONCURRENT="${N_CONCURRENT:-16}"
export MEMORY_MB="${MEMORY_MB:-2048}"
export RESUME_PASSES="${RESUME_PASSES:-2}"
# Re-run trials that ended in an infrastructure error; never the model's own failures
# (ResponseTruncatedError, EmptySolutionError), which would re-sample the model.
export RESUME_FILTER_ERRORS="${RESUME_FILTER_ERRORS:-AgentTimeoutError NonZeroAgentExitCodeError RuntimeError CancelledError NetworkConnectionError HTTPStatusError APITimeoutError}"
echo "tasks=${TASKS} model=${MODEL:-} agent=${AGENT:-mini-swe-agent} port=${VLLM_PORT} n=${N_CONCURRENT} resume_passes=${RESUME_PASSES}"

bash scripts/eval/run.sbatch
