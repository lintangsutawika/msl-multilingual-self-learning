#!/usr/bin/env bash
# Harbor verifier for {comp_id} (MLE-Bench). Grade the agent's predictions.csv
# against the prepared private answers via `mlebench grade`.
set -euo pipefail
SUBMISSION="/workspace/predictions.csv"
COMPETITION="{comp_id}"
# Neutralize Kaggle's interactive rule-acceptance prompts (headless -> EOF).
python3 /workspace/neutralize_rules.py
mlebench prepare -c "$COMPETITION"
WORKDIR=$(mktemp -d)
printf '{"competition_id":"%s","submission_path":"%s"}\n' "$COMPETITION" "$SUBMISSION" > "$WORKDIR/subs.jsonl"
mlebench grade "$WORKDIR/subs.jsonl" --output-dir "$WORKDIR/out" >/dev/null 2>&1 || true
REPORT=$(ls "$WORKDIR/out"/*grading_report*.json 2>/dev/null | head -1)
SCORE=$(python3 -c "import json,sys
try:
    rep=json.load(open(sys.argv[1]))
    c=rep.get('competition_reports') or []
    if c: print(c[0].get('score',''))
except Exception: pass
" "$REPORT" 2>/dev/null)
[ -z "$SCORE" ] && SCORE=1.0
# reward = 1/(1+score); score=0 -> 1.0, large -> ~0 (lower-better metrics assumed).
REWARD=$(python3 -c "print('%.5f' % (1.0/(1.0+float('$SCORE'))))")
echo "$REWARD" > /logs/verifier/reward.txt
echo "ok" > /logs/verifier/status.txt
echo "score=$SCORE reward=$REWARD"
rm -rf "$WORKDIR"
