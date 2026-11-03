#!/usr/bin/env python3
"""Grade a submission on ONE split of the test rows and print one JSON line. Never raises.

Shared by every task: copied next to the task's grade.py (in the box for `validate`, in tests/ for the verifier).
The task's grade.py follows MLE-bench's convention: `grade(submission: DataFrame, answers: DataFrame) -> float`,
raising on an invalid submission, plus a module constant HIGHER_IS_BETTER.

The agent predicts every test row; only the rows whose id is in --answers (the public or the private split) are
passed to grade(). Rows for other ids are ignored; a missing id makes the submission invalid.

    grade_cli.py --grader-dir DIR --answers split.csv --submission sub.csv [--id-column ID]
                 [--leaderboard lb.csv | --baseline S0 --ceiling S1]

Output: {"valid", "score", "higher_is_better", "rows", "error", "position_score", "normalized_score", "reward"}
- position_score: fraction of the leaderboard beaten or tied (MLE-Dojo's position score), if --leaderboard.
- normalized_score: (score - baseline) / (ceiling - baseline) clipped to [0, 1], direction-aware, if given.
- reward: position_score if available, else normalized_score, else 1.0 for any valid submission; 0 if invalid.
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import pandas as pd


def load_grader(d: Path):
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("task_grade", d / "grade.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def position(score: float, lb: str, hib: bool) -> float | None:
    try:
        d = pd.read_csv(lb)
        s = pd.to_numeric(d["score"] if "score" in d.columns else d.iloc[:, 1], errors="coerce").dropna().values
    except Exception:
        return None
    if len(s) == 0:
        return None
    return float((s <= score).sum() if hib else (s >= score).sum()) / len(s)


def normalized(score: float, base: float | None, ceil: float | None) -> float | None:
    if base is None or ceil is None or base == ceil:
        return None
    x = (score - base) / (ceil - base)  # ceiling is better than baseline in the metric's direction, so no sign flip
    return float(min(1.0, max(0.0, x)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grader-dir", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--id-column")
    ap.add_argument("--leaderboard")
    ap.add_argument("--baseline", type=float)
    ap.add_argument("--ceiling", type=float)
    a = ap.parse_args()
    out = {"valid": False, "score": None, "higher_is_better": None, "rows": None, "error": None,
           "position_score": None, "normalized_score": None, "reward": 0.0}
    try:
        g = load_grader(Path(a.grader_dir))
        hib = bool(g.HIGHER_IS_BETTER)
        out["higher_is_better"] = hib
        ans = pd.read_csv(a.answers)
        idc = a.id_column or ans.columns[0]
        out["rows"] = len(ans)
        try:
            sub = pd.read_csv(a.submission)
        except Exception as e:
            raise ValueError(f"cannot read submission as CSV: {e}")
        if idc not in sub.columns:
            raise ValueError(f"submission has no id column {idc!r} (columns: {list(sub.columns)[:10]})")
        want = set(ans[idc].astype(str))
        have = sub[idc].astype(str)
        missing = len(want - set(have))
        if missing:
            raise ValueError(f"submission is missing {missing} of the {len(want)} required ids")
        sub = sub[have.isin(want)].reset_index(drop=True)
        score = float(g.grade(sub.copy(), ans.copy()))
        if not math.isfinite(score):
            raise ValueError(f"score is {score}")
        out.update(valid=True, score=score)
        if a.leaderboard and Path(a.leaderboard).exists():
            out["position_score"] = position(score, a.leaderboard, hib)
        out["normalized_score"] = normalized(score, a.baseline, a.ceiling)
        for k in ("position_score", "normalized_score"):
            if out[k] is not None:
                out["reward"] = out[k]
                break
        else:
            out["reward"] = 1.0
    except Exception as e:  # invalid submission or grader bug: reward 0 with a readable reason
        out["error"] = f"{type(e).__name__}: {e}"[:500]
    print(json.dumps(out))


if __name__ == "__main__":
    main()
