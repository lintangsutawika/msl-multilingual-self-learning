#!/usr/bin/env bash
# Build the MLE training set end to end (CPU only): clone MLE-Dojo, download, import, split + checks, Harbor tasks,
# train.jsonl. Resumable: rerun after a preemption or after accepting more Kaggle rules.
# Works with sbatch or plain bash. Run from the repo root.
#
#   WORKDIR=~/mle-train bash src/benchmarks/mle_train/slurm/build.sbatch
#   sbatch --export=ALL,WORKDIR=$HOME/mle-train [cluster flags] src/benchmarks/mle_train/slurm/build.sbatch
#     e.g. CMU Babel: -A teruko -p preempt -q preempt_cpu_qos
#
# Env: WORKDIR (required), PYTHON (a Python >= 3.11 with requirements.txt installed; default python3),
#      EXTRA_ARGS (passed to `pipeline.py all`, e.g. "--no-kaggle" or "--max-gb 2").
#SBATCH --job-name=mle-train-build
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
: "${WORKDIR:?set WORKDIR to the output directory}"
"${PYTHON:-python3}" src/benchmarks/mle_train/pipeline.py all --workdir "$WORKDIR" ${EXTRA_ARGS:-} < /dev/null
