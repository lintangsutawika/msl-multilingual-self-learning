#!/usr/bin/env bash
# Final grading of /workspace/predictions.csv on the TEST set (MLE-Dojo's test rows, /tests/private/answers.csv),
# and on the validation rows too (/workspace/private_val): val_score is the final submission on the scale of
# `validate`, which the trainer compares with the validate scores. Never crashes: invalid/missing -> 0.
# Writes grade.json (the MLE-bench template's format, read by the trainer), reward.txt (leaderboard position),
# final_score.json (the grader's output) and predictions.csv.gz (the final predictions, so they can be regraded).
set -uo pipefail
mkdir -p /logs/verifier
SUB=/workspace/predictions.csv
grade() {  # grade <answers.csv> [extra grade_cli args...]
  python3 -I /tests/grader/grade_cli.py --grader-dir /tests/grader --answers "$1" --submission "$SUB" \
          --id-column '{id_column}' "${@:2}" 2>&1
}
if [ -f "$SUB" ]; then
  OUT=$(grade /tests/private/answers.csv --leaderboard /tests/leaderboard.csv)
  VAL=$(grade /workspace/private_val/answers.csv)
  gzip -c "$SUB" > /logs/verifier/predictions.csv.gz 2>/dev/null || true
else
  OUT='{"valid": false, "score": null, "error": "no /workspace/predictions.csv", "reward": 0.0}'
  VAL='{"valid": false, "score": null}'
fi
python3 - "$OUT" "$VAL" <<'PY'
import json, sys

def last_json(text):
    try:
        return json.loads(text.strip().splitlines()[-1])
    except Exception:
        return {"valid": False, "score": None, "reward": 0.0, "error": "grader crashed: " + text[-300:]}

r, v = last_json(sys.argv[1]), last_json(sys.argv[2])
json.dump(r, open("/logs/verifier/final_score.json", "w"))
valid = bool(r.get("valid"))
reward = float(r.get("reward") or 0.0) if valid else 0.0
# grade.json in the MLE-bench template's format: the trainer (src/trainer/spool_rollout.py) reads score,
# valid_submission and medal, and val_score / val_valid (its _final_val).
grade = {"score": r.get("score"), "valid_submission": valid, "above_median": bool(r.get("above_median")),
         "gold_threshold": r.get("gold_threshold"), "silver_threshold": r.get("silver_threshold"),
         "bronze_threshold": r.get("bronze_threshold"), "median_threshold": r.get("median_threshold"),
         "is_lower_better": None if r.get("higher_is_better") is None else not r["higher_is_better"],
         "medal": (r.get("medal") or "none") if valid else "none", "position_score": r.get("position_score"),
         "reward": reward, "val_score": v.get("score") if v.get("valid") else None, "val_valid": bool(v.get("valid"))}
if not valid:
    grade["error"] = r.get("error")
if not v.get("valid") and v.get("error"):
    grade["val_error"] = str(v.get("error"))[:300]
json.dump(grade, open("/logs/verifier/grade.json", "w"), indent=1)
open("/logs/verifier/reward.txt", "w").write(f"{reward:.5f}\n")
print(f"test score={r.get('score')} valid={valid} medal={grade['medal']} reward={reward:.5f} "
      f"val_score={grade['val_score']} error={r.get('error')}")
PY
