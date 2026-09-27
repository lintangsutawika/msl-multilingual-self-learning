#!/usr/bin/env bash
# Build one LeetCode image per language from the task templates' Dockerfiles
# (src/benchmarks/leetcode/task-template-<lang>/environment/Dockerfile), the same
# definitions `--prebuild-sif` uses, into $LEETCODE_IMAGE_DIR/leetcode-<lang>.sif.
#
# Usage: scripts/build/build_leetcode_images.sh [language ...]   (default: all 9)
set -euo pipefail

cd "$(dirname "$0")/../.."

IMAGE_DIR="${LEETCODE_IMAGE_DIR:-/data/user_data/$USER/msl-images}"
CONTAINER_BIN="${CONTAINER_BIN:-$(command -v apptainer || command -v singularity)}"

if [[ $# -gt 0 ]]; then
  LANGUAGES=("$@")
else
  LANGUAGES=(python cpp go java rust javascript typescript php ruby)
fi

mkdir -p "$IMAGE_DIR"

for lang in "${LANGUAGES[@]}"; do
  dockerfile="src/benchmarks/leetcode/task-template-$lang/environment/Dockerfile"
  sif_file="$IMAGE_DIR/leetcode-$lang.sif"
  [[ -f "$dockerfile" ]] || { echo "ERROR: missing $dockerfile" >&2; exit 1; }

  echo "=== Building $lang: $dockerfile -> $sif_file"
  # Build next to the target, then replace it in one step, so rebuilding works
  # and nothing reading the old image sees a partial file.
  tmp_sif="$sif_file.building"
  rm -f "$tmp_sif" "$tmp_sif.def"
  uv run python -c "
import sys
from pathlib import Path
from src.benchmarks.leetcode.adapter import prebuild_sif
prebuild_sif(Path(sys.argv[1]), Path(sys.argv[2]), container_bin=sys.argv[3])
" "$dockerfile" "$tmp_sif" "$CONTAINER_BIN"
  mv -f "$tmp_sif" "$sif_file"
  rm -f "$tmp_sif.def"
done

echo "Built: ${LANGUAGES[*]} in $IMAGE_DIR"
