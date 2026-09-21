#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

IMAGE_DIR="${LEETCODE_IMAGE_DIR:-/data/user_data/$USER/msl-images}"
DEF_DIR="src/benchmarks/leetcode/images"

LANGUAGES=(
  python
  cpp
  go
  java
  rust
  javascript
  typescript
  php
  ruby
)

mkdir -p "$IMAGE_DIR"

echo "Building LeetCode SIF images"
echo "Definition directory: $DEF_DIR"
echo "Output directory:     $IMAGE_DIR"
echo

for lang in "${LANGUAGES[@]}"; do
  def_file="$DEF_DIR/$lang.def"
  sif_file="$IMAGE_DIR/leetcode-$lang.sif"

  if [[ ! -f "$def_file" ]]; then
    echo "ERROR: missing definition file: $def_file" >&2
    exit 1
  fi

  echo "============================================================"
  echo "Building $lang"
  echo "  $def_file"
  echo "  -> $sif_file"
  echo "============================================================"

  apptainer build "$sif_file" "$def_file"

  echo
done

echo "All images built successfully:"
for lang in "${LANGUAGES[@]}"; do
  echo "  $IMAGE_DIR/leetcode-$lang.sif"
done
