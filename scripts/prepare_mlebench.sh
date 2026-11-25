#!/usr/bin/env bash
# Download MLE-bench competition data into MLE-bench's data dir. Download only: building tasks from it is
# scripts/mlebench_tasks.sh.
#
#   scripts/prepare_mlebench.sh lite                    # every competition of a set (a folder on the Drive)
#   scripts/prepare_mlebench.sh lite spooky-author-identification leaf-classification   # some of them
#   scripts/prepare_mlebench.sh lite --data /big/disk/mlebench-data   # data dir (default ~/.cache/mle-bench/data)
#   scripts/prepare_mlebench.sh medium --kaggle --data /big/disk/mlebench-data   # from Kaggle instead of the Drive
#
# The Drive holds MLE-bench's *prepared* data: what `mlebench prepare` makes from Kaggle's download (Kaggle's train
# split into train / test, with the test answers apart). One tar per competition, <set>/<comp>.tar, holding
# <comp>/public/ (what the agent gets) and <comp>/private/ (the test answers). Each tar is streamed and unpacked
# into <data>/<comp>/prepared/{public,private}, the layout `mlebench prepare` writes, so nothing needs Kaggle.
#
# --kaggle (medium and high: too big for the Drive) makes the same <data>/<comp>/prepared/ from Kaggle: per
# competition, Kaggle's zip is downloaded and unpacked in a scratch dir, MLE-bench's prepare.py (pinned revision, the
# prepare environment src/benchmarks/mlebench/prepare) writes prepared/ to the data dir, and the scratch dir is
# deleted. Without competitions, the set's list (MLE-bench's split file). Needs a Kaggle API key
# (~/.kaggle/kaggle.json, mode 600, or KAGGLE_USERNAME + KAGGLE_KEY) of an account that accepted each competition's
# rules. A competition that fails (rules not accepted, disk full, ...) is reported and the rest go on; rerun to retry
# (finished ones are skipped). Space: the scratch dir needs about 2.2 x the competition's zip (the largest, contrails,
# is a 325 GB zip); the data dir about 1 x the zip per competition (medium + high: ~3 TB in all). Big ones take hours:
# run it in a CPU job, e.g.
#   sbatch -p <cpu partition> -c 4 --mem=64G -t 2-00:00:00 --wrap "scripts/prepare_mlebench.sh high --kaggle --data DIR"
#
# Env:
#   DRIVE          rclone path of the shared folder (default gdrive:mlebench-prepared). If the folder was shared
#                  with you, it is under "Shared with me": DRIVE="gdrive,shared_with_me:mlebench-prepared"
#   MLEBENCH_DATA  same as --data
#   SCRATCH        --kaggle: where downloads are unpacked (default /scratch/$USER if it exists, else $TMPDIR or /tmp)
set -euo pipefail
DATA=${MLEBENCH_DATA:-$HOME/.cache/mle-bench/data}
KAGGLE=0
args=()
while [ $# -gt 0 ]; do
    case $1 in
        --data) DATA=$2; shift 2 ;;
        --data=*) DATA=${1#--data=}; shift ;;
        --kaggle) KAGGLE=1; shift ;;
        *) args+=("$1"); shift ;;
    esac
done
set -- "${args[@]}"
SET=${1:?usage: $0 SET [COMPETITION ...] [--data DIR] [--kaggle]}; shift
DATA=$(realpath -m "$DATA")
mkdir -p "$DATA"

if [ "$KAGGLE" = 1 ]; then
    case $SET in lite) LEVEL=low ;; medium|high) LEVEL=$SET ;; *) echo "unknown set $SET (lite, medium, high)" >&2; exit 2 ;; esac
    if [ ! -f "$HOME/.kaggle/kaggle.json" ] && { [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; }; then
        echo "no Kaggle API key: put it in ~/.kaggle/kaggle.json (chmod 600) or set KAGGLE_USERNAME and KAGGLE_KEY" >&2
        exit 2
    fi
    if [ -z "${SCRATCH:-}" ]; then
        if [ -d /scratch ] && mkdir -p "/scratch/$USER" 2>/dev/null; then SCRATCH=/scratch/$USER; else SCRATCH=${TMPDIR:-/tmp}; fi
    fi
    cd "$(dirname "$0")/.."
    uv sync -q --project src/benchmarks/mlebench/prepare
    PY=(uv run -q --project src/benchmarks/mlebench/prepare)
    if [ $# -gt 0 ]; then
        comps=("$@")
    else
        splits=$("${PY[@]}" python -c 'from src.benchmarks.mlebench.main import splits_dir; print(splits_dir())')
        mapfile -t comps < <(grep -v '^\s*$' "$splits/$LEVEL.txt")
    fi
    echo "${#comps[@]} competitions from Kaggle -> $DATA (scratch: $SCRATCH)"
    work=""
    trap '[ -n "$work" ] && rm -rf "${work:?}"' EXIT
    failed=()
    for c in "${comps[@]}"; do
        dst="$DATA/$c/prepared"
        if [ -f "$dst/.complete" ]; then echo "skip $c: already downloaded"; continue; fi
        echo "[$(date +%H:%M)] $c"
        work=$(mktemp -d "$SCRATCH/mlebench-$c.XXXXXX")
        if "${PY[@]}" kaggle competitions download -c "$c" -p "$work" </dev/null 2>&1 \
                | grep -v -i outdated | tr '\r' '\n' | grep -v '^\s*$' | tail -2 \
            && zip=$(ls "$work"/*.zip 2>/dev/null | head -1) && [ -n "$zip" ] \
            && echo "  unpack $(du -h "$zip" | cut -f1)" \
            && "${PY[@]}" python -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$zip" "$work/raw" \
            && rm -f "$zip" \
            && echo "  prepare" \
            && "${PY[@]}" python - "$c" "$work/raw" "$DATA" <<'PY'
import shutil, sys
from pathlib import Path
from mlebench.registry import registry
comp = registry.set_data_dir(Path(sys.argv[3])).get_competition(sys.argv[1])
for d in (comp.public_dir, comp.private_dir):  # <data>/<comp>/prepared/{public,private}
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
comp.prepare_fn(Path(sys.argv[2]), comp.public_dir, comp.private_dir)
(comp.public_dir / "description.md").write_text(comp.description)  # as `mlebench prepare` does
PY
        then
            touch "$dst/.complete"   # marks a finished download (an interrupted one is redone)
            echo "  $(du -sh "$dst" | cut -f1)  $dst"
        else
            echo "FAILED $c (rules not accepted on Kaggle? disk full? see above)" >&2
            failed+=("$c")
        fi
        rm -rf "${work:?}"; work=""
    done
    if [ ${#failed[@]} -gt 0 ]; then echo "done; ${#failed[@]} failed (rerun to retry): ${failed[*]}"; exit 1; fi
    echo "done"
    exit 0
fi

DRIVE=${DRIVE:-gdrive:mlebench-prepared}
if [ $# -gt 0 ]; then comps=("$@"); else mapfile -t comps < <(rclone lsf "$DRIVE/$SET" --include '*.tar' | sed 's/\.tar$//'); fi
[ ${#comps[@]} -gt 0 ] || { echo "no competitions found in $DRIVE/$SET" >&2; exit 1; }
echo "${#comps[@]} competitions from $DRIVE/$SET -> $DATA"
for c in "${comps[@]}"; do
    dst="$DATA/$c/prepared"
    if [ -f "$dst/.complete" ]; then echo "skip $c: already downloaded"; continue; fi
    echo "[$(date +%H:%M)] $c"
    tmp="$DATA/$c/.download"
    rm -rf "$tmp" && mkdir -p "$tmp"
    rclone cat "$DRIVE/$SET/$c.tar" | tar -x -C "$tmp"   # streamed: the tar is never stored
    [ -d "$tmp/$c/public" ] && [ -d "$tmp/$c/private" ] || { echo "FAILED $c: tar has no $c/public and $c/private" >&2; exit 1; }
    rm -rf "$dst" && mv "$tmp/$c" "$dst" && rmdir "$tmp"
    touch "$dst/.complete"   # marks a finished download (an interrupted one is redone)
    echo "  $(du -sh "$dst" | cut -f1)  $dst"
done
echo "done"
