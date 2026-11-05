#!/usr/bin/env python3
"""Score a submission on the public leaderboard (the hidden validation split of the test rows).

Usage:  validate <file.csv>

Runs as root (via /usr/local/bin/validate-root) for the unprivileged agent: grades a private copy of the
given CSV with grade_cli.py against /opt/grader/public_answers.csv, logs the score to
/logs/verifier/validation_scores.jsonl and prints `score=<s> sig=<hmac>`. An invalid submission prints only an
error (exit 1) and logs nothing, as in the MLE-bench template (src/benchmarks/mlebench/), so it does not close
an episode.

The signature is HMAC-SHA256(score_key, "<task_id>:<s>")[:16]; the rollout forwarder recomputes it with
the task's environment/score_key and ignores unsigned or wrongly signed scores, so an agent printing
`score=...` itself gets nothing.
"""
import hashlib
import hmac
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile

TASK_ID = "{task_id}"
ID_COLUMN = "{id_column}"
GRADER = "/opt/grader"
KEY_FILE = "/etc/msl/score_key"
LOG = "/logs/verifier/validation_scores.jsonl"
# Never grade a file from these trees (grading a root-only file as a "submission" could leak or fake a score).
PROTECTED = (GRADER, "/opt/msl", "/etc", "/root", "/proc", "/sys", "/dev", "/tests", "/logs")


def checked_copy(path: str, work: str) -> str:
    """Copy an agent-owned regular file into work/; refuse anything else."""
    real = os.path.realpath(path)
    if any(real == p or real.startswith(p + "/") for p in PROTECTED):
        raise ValueError("refusing to grade a protected file")
    st = os.stat(real)
    if not stat.S_ISREG(st.st_mode) or st.st_nlink > 1:
        raise ValueError("submission must be a regular file (no links)")
    uid = os.environ.get("SUDO_UID")
    if uid is not None and st.st_uid != int(uid):
        raise ValueError("submission must be a file you created")
    dst = os.path.join(work, "submission.csv")
    shutil.copyfile(real, dst)  # grade the copy: the file can't change under us
    return dst


def grade(submission: str) -> dict:
    """grade_cli.py's JSON result: valid, score, higher_is_better, error, ..."""
    r = subprocess.run(
        [sys.executable, "-I", GRADER + "/grade_cli.py", "--grader-dir", GRADER,
         "--answers", GRADER + "/public_answers.csv", "--submission", submission, "--id-column", ID_COLUMN],
        capture_output=True, text=True,
    )
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        tail = (r.stdout + r.stderr)[-300:]
        return {"valid": False, "score": None, "higher_is_better": None, "error": "grader crashed: " + tail}


def sign(score_text: str) -> str:
    key = open(KEY_FILE, "rb").read().strip()
    return hmac.new(key, (TASK_ID + ":" + score_text).encode(), hashlib.sha256).hexdigest()[:16]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    sub_path = sys.argv[1]
    with tempfile.TemporaryDirectory() as work:
        try:
            r = grade(checked_copy(sub_path, work))
        except Exception as e:
            r = {"valid": False, "error": str(e)}
    if not r.get("valid"):
        print("validate: invalid submission: " + str(r.get("error")), file=sys.stderr)
        return 1
    score = float(r["score"])
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps({"submission_path": sub_path, "submission_score": score}) + "\n")
    # str(float) always matches the forwarder's score regex (e.g. 0.8123, 1e-05, -3.5).
    score_text = str(score)
    print("score=" + score_text + " sig=" + sign(score_text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
