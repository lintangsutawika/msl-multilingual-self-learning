#!/usr/bin/env bash
# Download MLE-bench competition data into MLE-bench's data dir. Download only: building tasks from it is
# scripts/mlebench_tasks.sh.
#
#   scripts/prepare_mlebench.sh lite                    # every competition of a set (a folder on the Drive)
#   scripts/prepare_mlebench.sh lite spooky-author-identification leaf-classification   # some of them
#   scripts/prepare_mlebench.sh lite --data /big/disk/mlebench-data   # data dir (default ~/.cache/mle-bench/data)
#
# The Drive holds MLE-bench's *prepared* data: what `mlebench prepare` makes from Kaggle's download (Kaggle's train
# split into train / test, with the test answers apart). One tar per competition, <set>/<comp>.tar, holding
# <comp>/public/ (what the agent gets) and <comp>/private/ (the test answers). Each tar is streamed and unpacked
# into <data>/<comp>/prepared/{public,private}, the layout `mlebench prepare` writes, so nothing needs Kaggle.
#
# Env:
#   DRIVE          rclone path of the shared folder (default gdrive:mlebench-prepared). If the folder was shared
#                  with you, it is under "Shared with me": DRIVE="gdrive,shared_with_me:mlebench-prepared"
#   MLEBENCH_DATA  same as --data
set -euo pipefail
DATA=${MLEBENCH_DATA:-$HOME/.cache/mle-bench/data}
args=()
while [ $# -gt 0 ]; do
    case $1 in
        --data) DATA=$2; shift 2 ;;
        --data=*) DATA=${1#--data=}; shift ;;
        *) args+=("$1"); shift ;;
    esac
done
set -- "${args[@]}"
SET=${1:?usage: $0 SET [COMPETITION ...] [--data DIR]}; shift
DRIVE=${DRIVE:-gdrive:mlebench-prepared}
mkdir -p "$DATA"
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
