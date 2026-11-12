#!/usr/bin/env bash
# Harbor verifier for {comp_id} (MLE-Bench). Grade the agent's final predictions.csv
# against the held-out answers via `mlebench grade` and report the medal it earns.
#
#   /logs/verifier/reward.txt   medal tier: 3 gold, 2 silver, 1 bronze, 0 none/invalid
#   /logs/verifier/grade.json   score, medal, thresholds, above_median, valid_submission
#
# This is the end-of-trial evaluation only; the PPO training reward comes from the
# signed `validate` scores recorded in the rollout spool.
set -euo pipefail
export PATH=/usr/local/bin:$PATH  # the image's Python (with mlebench), not a system one
SUBMISSION="/workspace/predictions.csv"
COMPETITION="{comp_id}"
mkdir -p /logs/verifier
# Neutralize Kaggle's interactive rule-acceptance prompts (harmless no-op).
python3 /opt/msl/neutralize_rules.py
# Test answers and leaderboard: tests/ (tasks built with --mount-data; uploaded only now) or baked into /workspace.
PRIVATE=/workspace/private; LEADERBOARD=/workspace/leaderboard.csv
[ -d /tests/private ] && PRIVATE=/tests/private
[ -f /tests/leaderboard.csv ] && LEADERBOARD=/tests/leaderboard.csv
# Place them where mlebench grade reads them (no Kaggle). Grading needs only the answers, not the public data.
python3 - "$COMPETITION" "$PRIVATE" "$LEADERBOARD" <<'PY'
import pathlib, shutil, sys
from mlebench.registry import registry
comp = registry.get_competition(sys.argv[1])
shutil.copytree(sys.argv[2], comp.private_dir, dirs_exist_ok=True)
comp.leaderboard.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(sys.argv[3], comp.leaderboard)
PY
WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT
# Validation split (src/benchmarks/mlebench/valsplit.py): predictions.csv also covers the validation rows; grade only
# MLE-bench's test rows, so the final score is MLE-bench's.
if [ -f /workspace/private_val/valsplit.json ] && [ -f "$SUBMISSION" ]; then
    python3 -c "import sys; sys.path.insert(0, '/opt/msl'); from grade_subset import subset_csv; subset_csv(*sys.argv[1:])" \
        "$SUBMISSION" "$COMPETITION" "$PRIVATE" "$WORKDIR/predictions.csv"
    SUBMISSION="$WORKDIR/predictions.csv"
fi
printf '{"competition_id":"%s","submission_path":"%s"}\n' "$COMPETITION" "$SUBMISSION" > "$WORKDIR/subs.jsonl"
mlebench grade --submission "$WORKDIR/subs.jsonl" --output-dir "$WORKDIR/out" >"$WORKDIR/grade.log" 2>&1 || true
python3 - "$WORKDIR" <<'PY'
import glob, json, sys
work = sys.argv[1]
rep = {}
g = glob.glob(f"{work}/out/*grading_report*.json")
if g:
    reps = json.load(open(g[0])).get("competition_reports") or []
    rep = reps[0] if reps else {}
medal = ("gold" if rep.get("gold_medal") else "silver" if rep.get("silver_medal")
         else "bronze" if rep.get("bronze_medal") else "none")
tier = {"gold": 3, "silver": 2, "bronze": 1, "none": 0}[medal]
out = {k: rep.get(k) for k in ("score", "valid_submission", "above_median", "gold_threshold",
                               "silver_threshold", "bronze_threshold", "median_threshold",
                               "is_lower_better")}
out.update(medal=medal, reward=tier)
if not rep:
    out["error"] = open(f"{work}/grade.log").read()[-500:]
json.dump(out, open("/logs/verifier/grade.json", "w"), indent=1)
open("/logs/verifier/reward.txt", "w").write(f"{tier}\n")
print(f"score={out['score']} medal={medal} reward={tier}")
PY
echo "ok" > /logs/verifier/status.txt
