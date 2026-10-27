#!/usr/bin/env python3
"""Validate a MLE-Bench submission file against the held-out private answers.

Usage:  validate <predictions.csv>

Grades the given CSV (any filename) with `mlebench grade`, prints the score,
and logs it to /logs/verifier/validation_scores.jsonl so the harness can reward
improvement across the run. Requires the data to be prepared (startup.sh does
this at container start).
"""
import json
import subprocess
import sys
import tempfile
import glob
import os

COMPETITION = "{comp_id}"


def grade(submission_path: str) -> float:
    os.environ.setdefault("KAGGLE_CONFIG_DIR", "/root/.kaggle")
    with tempfile.TemporaryDirectory() as work:
        subs = f"{work}/subs.jsonl"
        with open(subs, "w") as f:
            f.write(json.dumps({"competition_id": COMPETITION,
                                "submission_path": submission_path}) + "\n")
        r = subprocess.run(
            ["mlebench", "grade", subs, "--output-dir", f"{work}/out"],
            capture_output=True, text=True,
        )
        g = glob.glob(f"{work}/out/*grading_report*.json")
        if g:
            rep = json.load(open(g[0]))
            comps = rep.get("competition_reports") or []
            if comps:
                return float(comps[0].get("score"))
        raise RuntimeError(f"grade produced no score; rc={r.returncode} {r.stdout[-200:]!r}")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    sub_path = sys.argv[1]
    score = grade(sub_path)
    os.makedirs("/logs/verifier", exist_ok=True)
    with open("/logs/verifier/validation_scores.jsonl", "a") as f:
        f.write(json.dumps({"submission_path": sub_path, "submission_score": score}) + "\n")
    print(f"score={score}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
