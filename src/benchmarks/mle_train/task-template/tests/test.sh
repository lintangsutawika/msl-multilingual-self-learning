#!/usr/bin/env bash
# Final grading on the PRIVATE leaderboard (the other ~70% of the test ids). Never crashes: invalid/missing -> 0.
set -uo pipefail
mkdir -p /logs/verifier
SUB=/workspace/predictions.csv
if [ -f "$SUB" ]; then
  OUT=$(python3 /tests/grader/grade_cli.py --grader-dir /tests/grader --answers /tests/grader/private_answers.csv \
        --submission "$SUB" --id-column '{id_column}' {reward_args} 2>&1)
else
  OUT='{"valid": false, "score": null, "error": "no /workspace/predictions.csv", "reward": 0.0}'
fi
python3 - "$OUT" <<'PY'
import json, sys
try:
    r = json.loads(sys.argv[1].strip().splitlines()[-1])
except Exception:
    r = {"valid": False, "score": None, "reward": 0.0, "error": "grader crashed: " + sys.argv[1][-300:]}
json.dump(r, open("/logs/verifier/final_score.json", "w"))
reward = float(r.get("reward") or 0.0) if r.get("valid") else 0.0
open("/logs/verifier/reward.txt", "w").write(f"{reward:.5f}\n")
print(f"private score={r.get('score')} valid={r.get('valid')} reward={reward:.5f} error={r.get('error')}")
PY
