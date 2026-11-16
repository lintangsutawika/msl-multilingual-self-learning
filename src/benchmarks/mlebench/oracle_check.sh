#!/usr/bin/env bash
# Check freshly built tasks before sweeping them: run the oracle (solution/solve.sh submits MLE-bench's sample
# submission, or the competition's equivalent) through run_babel.sh, then show what each grader made of it:
#   validate  the host-side validation grader's score (tasks built with --val-split), from <job>.validate.jsonl
#   test      the final MLE-bench grade (verifier/grade.json): valid_submission and score
# A sample submission scores near chance; what matters is that both graders accept it. Exit status 1 if any task's
# final grade is not a valid submission. Run on a compute node (Singularity), from the repo root:
#
#   sbatch --cpus-per-task=4 --mem=16G --time=3:00:00 --output=logs/%x-%j.out \
#       --wrap "src/benchmarks/mlebench/oracle_check.sh /path/to/tasks/mlebench-low/<comp> ..."
#
# Env: JOBS_DIR (default <task parent>/../oracle); site paths from site.env (run_babel.sh).
set -uo pipefail
cd "$(dirname "$0")/../../.."
rc=0
for TASK in "$@"; do
    TASK=$(realpath "$TASK"); NAME=$(basename "$TASK")
    J=${JOBS_DIR:-$(dirname "$(dirname "$TASK")")/oracle}
    AGENT=oracle JOBS_DIR="$J" JOB_PREFIX=oracle src/benchmarks/mlebench/run_babel.sh "$TASK"
    d=$(ls -d "$J"/oracle-"$NAME"-* 2>/dev/null | grep -v '\.jsonl$' | tail -1)
    g=$(ls "$d"/*/verifier/grade.json 2>/dev/null | head -1)
    echo "$NAME validate: $(tail -1 "$d.validate.jsonl" 2>/dev/null || echo '(no validation split)')"
    if [ -n "$g" ] && python3 - "$g" "$NAME" <<'PY'
import json, sys
g = json.load(open(sys.argv[1]))
print(f"{sys.argv[2]} test: valid={g.get('valid_submission')} score={g.get('score')}")
sys.exit(0 if g.get("valid_submission") else 1)
PY
    then echo "ok   $NAME"; else echo "FAIL $NAME (grade: ${g:-none})"; rc=1; fi
done
exit $rc
