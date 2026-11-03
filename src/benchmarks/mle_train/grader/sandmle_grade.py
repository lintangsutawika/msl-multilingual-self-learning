"""grade.py for tasks regenerated from SandMLE: runs the task's own evaluator.py on a split.

SandMLE evaluators are standalone CLIs (`--submission_path`, optionally `--answer_path`, else they read
answer.csv from the working directory) that print JSON with "score" (None on failure) and "is_lower_better".
Copied into <task>/grader/ next to evaluator.py; HIGHER_IS_BETTER is filled in at import time.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALUATOR = HERE / "evaluator.py"
HIGHER_IS_BETTER = __HIGHER_IS_BETTER__


def _last_json(text: str) -> dict:
    dec, found, i = json.JSONDecoder(), None, text.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(text, i)
            if isinstance(obj, dict) and "score" in obj:
                found = obj
            i = text.find("{", end)
        except ValueError:
            i = text.find("{", i + 1)
    if found is None:
        raise ValueError("evaluator printed no result: " + text[-300:])
    return found


def grade(submission, answers) -> float:
    with tempfile.TemporaryDirectory() as d:
        sub, ans = Path(d) / "submission.csv", Path(d) / "answer.csv"
        submission.to_csv(sub, index=False)
        answers.to_csv(ans, index=False)
        cmd = [sys.executable, str(EVALUATOR), "--submission_path", str(sub)]
        if "--answer_path" in EVALUATOR.read_text():
            cmd += ["--answer_path", str(ans)]
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=1800)
        res = _last_json(r.stdout)
    if res.get("score") is None:
        raise ValueError(str(res.get("error") or "evaluator returned no score")[:400])
    return float(res["score"])
