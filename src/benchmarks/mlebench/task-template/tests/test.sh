#!/usr/bin/env bash
# Harbor verifier for {comp_id} (MLE-Bench). Grade the agent's predictions.csv
# against the prepared private answers via `mlebench grade`.
set -euo pipefail
SUBMISSION="/workspace/predictions.csv"
COMPETITION="{comp_id}"
# Neutralize Kaggle's interactive rule-acceptance prompts (harmless no-op).
python3 /workspace/neutralize_rules.py
# Data is BAKED into the image; place it where mlebench grade reads them (no Kaggle).
python3 - <<'PY'
from mlebench.registry import registry
import shutil, pathlib
comp = registry.get_competition("$COMPETITION")
comp.private_dir.mkdir(parents=True, exist_ok=True)
comp.public_dir.parent.mkdir(parents=True, exist_ok=True)
if pathlib.Path('/workspace/private').exists():
    shutil.copytree('/workspace/private', comp.private_dir, dirs_exist_ok=True)
if pathlib.Path('/workspace/public').exists():
    shutil.copytree('/workspace/public', comp.public_dir, dirs_exist_ok=True)
lb = comp.leaderboard
lb.parent.mkdir(parents=True, exist_ok=True)
shutil.copy('/workspace/leaderboard.csv', lb)
print('[verifier] baked private/public/leaderboard placed for mlebench grade')
PY
WORKDIR=$(mktemp -d)
printf '{"competition_id":"%s","submission_path":"%s"}\n' "$COMPETITION" "$SUBMISSION" > "$WORKDIR/subs.jsonl"
mlebench grade --submission "$WORKDIR/subs.jsonl" --output-dir "$WORKDIR/out" >/dev/null 2>&1 || true
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
