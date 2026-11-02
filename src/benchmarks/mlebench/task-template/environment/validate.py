#!/usr/bin/env python3
"""Validate a MLE-Bench submission file against the held-out private answers.

Usage:  validate <predictions.csv>

Runs as root (via /usr/local/bin/validate-root) for the unprivileged agent:
grades a private copy of the given CSV with `mlebench grade`, prints the score
with an HMAC signature (`score=<s> sig=<hex>`) that node B checks before using it
as reward, and logs it to /logs/verifier/validation_scores.jsonl. Requires the
data to be prepared (startup.sh does this at container start).
"""
import glob
import hashlib
import hmac
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile

COMPETITION = "{comp_id}"
KEY_FILE = "/etc/msl/score_key"
# Never grade a file from these trees: grading the answers themselves (or any
# root-owned file) as a "submission" would yield a perfect, validly signed score.
PROTECTED = ("/workspace/private", "/opt/msl", "/etc", "/root", "/proc", "/sys", "/dev")


def checked_copy(path: str, work: str) -> str:
    """Copy an agent-owned regular file into work/; refuse anything else."""
    real = os.path.realpath(path)
    if any(real == p or real.startswith(p + "/") for p in PROTECTED):
        sys.exit("validate: refusing to grade a protected file")
    st = os.stat(real)
    if not stat.S_ISREG(st.st_mode) or st.st_nlink > 1:
        sys.exit("validate: submission must be a regular file")
    uid = os.environ.get("SUDO_UID")
    if uid is not None and st.st_uid != int(uid):
        sys.exit("validate: submission must be a file you created")
    dst = os.path.join(work, "submission" + os.path.splitext(real)[1])
    shutil.copyfile(real, dst)  # grade the copy: the file can't change under us
    return dst


def grade(submission_path: str, work: str) -> float:
    os.environ.setdefault("KAGGLE_CONFIG_DIR", "/root/.kaggle")
    subs = f"{work}/subs.jsonl"
    with open(subs, "w") as f:
        f.write(json.dumps({"competition_id": COMPETITION,
                            "submission_path": submission_path}) + "\n")
    r = subprocess.run(
        ["mlebench", "grade", "--submission", subs, "--output-dir", f"{work}/out"],
        capture_output=True, text=True,
    )
    g = glob.glob(f"{work}/out/*grading_report*.json")
    if g:
        rep = json.load(open(g[0]))
        comps = rep.get("competition_reports") or []
        if comps and comps[0].get("score") is not None:
            return float(comps[0].get("score"))
    raise RuntimeError(f"grade produced no score; rc={r.returncode} {r.stdout[-200:]!r}")


def sign(score_text: str) -> str | None:
    try:
        key = open(KEY_FILE, "rb").read().strip()
    except OSError:
        return None
    return hmac.new(key, f"{COMPETITION}:{score_text}".encode(), hashlib.sha256).hexdigest()[:16]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    sub_path = sys.argv[1]
    with tempfile.TemporaryDirectory() as work:
        score = grade(checked_copy(sub_path, work), work)
    os.makedirs("/logs/verifier", exist_ok=True)
    with open("/logs/verifier/validation_scores.jsonl", "a") as f:
        f.write(json.dumps({"submission_path": sub_path, "submission_score": score}) + "\n")
    score_text = f"{score}"
    sig = sign(score_text)
    print(f"score={score_text}" + (f" sig={sig}" if sig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
