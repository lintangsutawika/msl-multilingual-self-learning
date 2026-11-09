#!/usr/bin/env bash
# Final grading on the PRIVATE leaderboard (the other half of the test ids). Never crashes: invalid/missing -> 0.
# Writes reward.txt (leaderboard position), final_score.json (grader output) and grade.json (medal, for the trainer).
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
# grade.json in the MLE-bench template's format: the trainer (src/trainer/spool_rollout.py) closes a trial's final
# episode with it and rewards the medal (gold 1.0, silver 0.67, bronze 0.33, valid without medal 0.1, invalid 0).
valid = bool(r.get("valid"))
grade = {"score": r.get("score"), "valid_submission": valid, "above_median": bool(r.get("above_median")),
         "gold_threshold": r.get("gold_threshold"), "silver_threshold": r.get("silver_threshold"),
         "bronze_threshold": r.get("bronze_threshold"), "median_threshold": r.get("median_threshold"),
         "is_lower_better": None if r.get("higher_is_better") is None else not r["higher_is_better"],
         "medal": (r.get("medal") or "none") if valid else "none", "position_score": r.get("position_score")}
if not valid:
    grade["error"] = r.get("error")
json.dump(grade, open("/logs/verifier/grade.json", "w"), indent=1)
open("/logs/verifier/reward.txt", "w").write(f"{reward:.5f}\n")
print(f"private score={r.get('score')} valid={r.get('valid')} medal={grade['medal']} reward={reward:.5f} "
      f"error={r.get('error')}")
PY
