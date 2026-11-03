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


def ensure_data() -> None:
    """Place the baked answers where `mlebench grade` reads them (as tests/test.sh does),
    so grading does not depend on startup.sh having run with the same HOME."""
    from mlebench.registry import registry

    comp = registry.get_competition(COMPETITION)
    for src, dst in (("/workspace/private", comp.private_dir), ("/workspace/public", comp.public_dir)):
        if os.path.isdir(src) and not (os.path.isdir(dst) and os.listdir(dst)):
            shutil.copytree(src, dst, dirs_exist_ok=True)
    # Always overwrite: the leaderboard bundled with a pip-installed mlebench can be a
    # placeholder without a `score` column (test.sh copies it unconditionally too).
    if os.path.exists("/workspace/leaderboard.csv"):
        os.makedirs(os.path.dirname(comp.leaderboard), exist_ok=True)
        shutil.copy("/workspace/leaderboard.csv", comp.leaderboard)


def grade(submission_path: str, work: str) -> float:
    os.environ.setdefault("KAGGLE_CONFIG_DIR", "/root/.kaggle")
    ensure_data()
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
    raise RuntimeError(f"grade produced no score; rc={r.returncode} "
                       f"stdout={r.stdout[-200:]!r} stderr={r.stderr[-300:]!r}")


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
