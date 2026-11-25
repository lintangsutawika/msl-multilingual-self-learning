#!/usr/bin/env bash
# Build MLE-bench Harbor tasks from downloaded data (scripts/prepare_mlebench.sh), with a validation split:
#
#   scripts/mlebench_tasks.sh lite                                   # every downloaded competition of a set
#   scripts/mlebench_tasks.sh lite spooky-author-identification ...  # some of them
#   scripts/mlebench_tasks.sh lite --data /big/disk/mlebench-data --mount-data  # Babel tasks (data mounted, not copied)
#
# For the competitions given (or every competition of the set whose data is downloaded):
#   1. valsplit.py      carve a validation set out of MLE-bench's train with the competition's own prepare.py
#                       (<data>/<comp>/valsplit/; done once, kept)
#   2. audit_leaks.py   check that no test or validation answer is visible to the agent; stops here on any leak
#   3. main.py          plain tasks:  $OUT/mlebench-<level>/<comp>        (validate grades the test answers)
#   4. make_split_tasks.py  split tasks: $OUT/mlebench-<level>-split/<comp>  (validate grades the validation set,
#                       the final grade is MLE-bench's test)
# Everything runs in the prepare environment (src/benchmarks/mlebench/prepare: MLE-bench pinned, pandas<3).
#
# Options: --data DIR (MLE-bench's data dir, as given to prepare_mlebench.sh; default ~/.cache/mle-bench/data or
# $MLEBENCH_DATA), --mount-data (Babel / Singularity: the data is mounted read-only at run time instead of copied into
# each task; run the tasks with src/benchmarks/mlebench/run_babel.sh), --resplit (redo the validation split even if one
# exists, e.g. after a split rule changed; seeded, so an unchanged rule gives the same split). Sets: lite (= MLE-bench's "low" split), medium,
# high. Env: OUT (default harbor_tasks). Splitting big competitions takes a while: run it in a CPU job.
set -euo pipefail
MOUNT=()
RESPLIT=()
DATA=${MLEBENCH_DATA:-$HOME/.cache/mle-bench/data}
args=()
while [ $# -gt 0 ]; do
    case $1 in
        --data) DATA=$2; shift 2 ;;
        --data=*) DATA=${1#--data=}; shift ;;
        --mount-data) MOUNT=(--mount-data); shift ;;
        --resplit) RESPLIT=(--force); shift ;;
        *) args+=("$1"); shift ;;
    esac
done
set -- "${args[@]}"
SET=${1:?usage: $0 SET [COMPETITION ...] [--data DIR] [--mount-data]}; shift
export MLEBENCH_DATA=$(realpath -m "$DATA")  # every tool below reads the data from here (src/benchmarks/mlebench/__init__.py)
OUT=$(realpath -m "${OUT:-harbor_tasks}")
cd "$(dirname "$0")/.."
uv sync -q --project src/benchmarks/mlebench/prepare
PY=(uv run -q --project src/benchmarks/mlebench/prepare python)
case $SET in lite) LEVEL=low ;; medium|high) LEVEL=$SET ;; *) echo "unknown set $SET (lite, medium, high)" >&2; exit 2 ;; esac

if [ $# -gt 0 ]; then
    comps=("$@")
    for c in "${comps[@]}"; do
        [ -d "$MLEBENCH_DATA/$c/prepared/public" ] \
            || { echo "$c: not downloaded to $MLEBENCH_DATA (scripts/prepare_mlebench.sh $SET $c --data ...)" >&2; exit 1; }
    done
else  # the set's competitions (MLE-bench's split file) whose data is downloaded
    splits=$("${PY[@]}" -c 'from src.benchmarks.mlebench.main import splits_dir; print(splits_dir())')
    DATA=$MLEBENCH_DATA
    comps=()
    for c in $(cat "$splits/$LEVEL.txt"); do
        if [ -d "$DATA/$c/prepared/public" ]; then comps+=("$c"); else echo "skip $c: not downloaded"; fi
    done
fi
[ ${#comps[@]} -gt 0 ] || { echo "no competitions to build" >&2; exit 1; }
C=$(IFS=,; echo "${comps[*]}")
echo "== ${#comps[@]} competitions -> $OUT"

echo "== 1. validation split"
"${PY[@]}" -m src.benchmarks.mlebench.valsplit --competition "$C" ${RESPLIT[@]+"${RESPLIT[@]}"}
echo "== 2. leak audit"
"${PY[@]}" -m src.benchmarks.mlebench.audit_leaks --competition "$C" \
    || { echo "leak audit failed: no tasks built (fix the split first)" >&2; exit 1; }
echo "== 3. plain tasks"
"${PY[@]}" -m src.benchmarks.mlebench --competition "$C" --output "$OUT" ${MOUNT[@]+"${MOUNT[@]}"}
echo "== 4. split tasks"
for c in "${comps[@]}"; do
    plain=$(ls -d "$OUT"/mlebench-*/"$c" | grep -v -- '-split/')   # main.py files it under the competition's level
    "${PY[@]}" src/benchmarks/mlebench/make_split_tasks.py --tasks "$c" --src "$(dirname "$plain")" \
        --dst "$(dirname "$plain")-split"
done
echo "== done: plain tasks in $OUT/mlebench-<level>/, split tasks in $OUT/mlebench-<level>-split/"
