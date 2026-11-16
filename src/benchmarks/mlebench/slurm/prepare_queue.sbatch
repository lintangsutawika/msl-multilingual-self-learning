#!/bin/bash
# Prepare MLE-bench competitions until the disk is full (src/benchmarks/mlebench/prepare_queue.py), with the grader
# venv (GRADER_PY in site.env). Resumable: requeue/resubmit after a preemption or after accepting more Kaggle rules;
# prepared competitions are skipped. Submit from the repo root:
#   sbatch src/benchmarks/mlebench/slurm/prepare_queue.sbatch                                  # every split
#   sbatch --export=ALL,EXTRA="--competition iwildcam-2019-fgvc6 --delete-zips" src/benchmarks/mlebench/slurm/prepare_queue.sbatch
# Env: SPLITS (default low,medium,high), RESERVE_GB (200), EXTRA (more prepare_queue flags), KAGGLE_CONFIG_DIR.
#SBATCH --job-name=mlebench-prepare
#SBATCH --account=teruko
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[ -f src/benchmarks/mlebench/site.env ] && . src/benchmarks/mlebench/site.env
: "${GRADER_PY:?set GRADER_PY (src/benchmarks/mlebench/site.env.example)}"
export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"
"$GRADER_PY" -m src.benchmarks.mlebench.prepare_queue --splits "${SPLITS:-low,medium,high}" \
    --reserve-gb "${RESERVE_GB:-200}" ${EXTRA:-}
