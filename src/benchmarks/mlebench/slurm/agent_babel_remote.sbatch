#!/usr/bin/env bash
# agent_babel.sbatch against a remote model server instead of a vLLM on the job's GPU: CPU only. A local proxy
# (src/relay/local_proxy.py) adds the server's key and the MAX_GEN_TOKENS default, so the agent, its settings and
# Harbor's job config are as in agent_babel.sbatch (127.0.0.1, key EMPTY). Same usage and env as agent_babel.sbatch,
# plus LLM_ENV_FILE (default ~/.config/msl/qwen9b.env, mode 600: <LLM_PREFIX>_BASE_URL, _API_KEY, _MODEL) and
# LLM_PREFIX (default QWEN9B). Resources: 4 attempts x (4 CPUs, 16 GB), as the GPU job.
#
#   sbatch --export=ALL,TASKS=/path/task,SWEEP_NAME=qwen3.5-9b-k4 src/benchmarks/mlebench/slurm/agent_babel_remote.sbatch
#SBATCH --job-name=mlebench-agent
#SBATCH --account=teruko
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=72G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out
export LLM_ENV_FILE="${LLM_ENV_FILE:-$HOME/.config/msl/qwen9b.env}" LLM_PREFIX="${LLM_PREFIX:-QWEN9B}"
[ -r "$LLM_ENV_FILE" ] || { echo "LLM_ENV_FILE $LLM_ENV_FILE not readable"; exit 1; }
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
exec bash src/benchmarks/mlebench/slurm/agent_babel.sbatch
