#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export LEETCODE_IMAGE_DIR="${LEETCODE_IMAGE_DIR:-/data/user_data/$USER/msl-images}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/$USER/uv-cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/scratch/$USER/apptainer-tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/scratch/$USER/apptainer-cache}"

mkdir -p \
  "$UV_CACHE_DIR" \
  "$APPTAINER_TMPDIR" \
  "$APPTAINER_CACHEDIR"

export TASK_PATH="${TASK_PATH:-benchmarks/leetcode/tasks-b1}"

exec scripts/eval/run.sh "$@"
