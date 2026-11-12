#!/bin/bash
# Prepare MLE-bench competitions until the disk is full (src/benchmarks/mlebench/prepare_queue.py). Resumable:
# requeue/resubmit after a preemption or after accepting more Kaggle rules; prepared competitions are skipped.
#   sbatch src/benchmarks/mlebench/slurm/prepare_queue.sbatch            # keeps Kaggle's zips
#   sbatch --export=ALL,EXTRA=--delete-zips src/benchmarks/mlebench/slurm/prepare_queue.sbatch
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
export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"
.venv/bin/python -m src.benchmarks.mlebench.prepare_queue --splits "${SPLITS:-low,medium,high}" \
    --reserve-gb "${RESERVE_GB:-200}" ${EXTRA:-}
