"""Sanity check of the validation splits (valsplit.py) with each task's own grader (grade_cli.py).

Per task:
  perfect   the validation answers as a submission, graded on the validation rows: must be valid and rank at the
            top of the Kaggle leaderboard (position >= MIN_PERFECT_POSITION), else the grader or the answers are off;
  val       the sample submission on the validation rows: must be valid;
  test      the sample submission on the test rows (the final grade), for comparison: a validation set drawn like
            the test set should put the sample submission at a similar leaderboard position.

    python -m src.benchmarks.mledojo.check_split --data DIR [--task a,b] [--workers 8] > check.tsv
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MIN_PERFECT_POSITION = 0.9
MAX_POSITION_GAP = 0.25  # |val - test| leaderboard position of the sample submission, beyond which we flag it


def _perfect(ans: pd.DataFrame, sample_cols: list[str], id_col: str) -> pd.DataFrame:
    """The true answers in submission format (one label column -> one-hot class columns where the sample wants them)."""
    if all(c in ans.columns for c in sample_cols):
        return ans[sample_cols]
    labels = [c for c in ans.columns if c != id_col]
    if len(labels) == 1:
        lab = ans[labels[0]].astype(str)
        out = pd.DataFrame({id_col: ans[id_col]})
        for c in sample_cols[1:]:
            hit = [v for v in lab.unique() if c.lower().endswith("_" + v.lower()) or c.lower() == v.lower()]
            if len(hit) != 1:
                return ans
            out[c] = (lab == hit[0]).astype(float)
        return out
    return ans


def _grade(task: Path, answers: Path, sub: Path, id_col: str) -> dict:
    cmd = [sys.executable, str(HERE / "grade_cli.py"), "--grader-dir", str(task / "grader"), "--answers", str(answers),
           "--submission", str(sub), "--id-column", id_col]
    if (task / "leaderboard.csv").exists():
        cmd += ["--leaderboard", str(task / "leaderboard.csv")]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"valid": False, "error": f"{type(e).__name__}: {e}"[:200]}


def check(task: Path) -> dict:
    import yaml

    vs = task / "valsplit"
    sample_name = yaml.safe_load((task / "config.yaml").read_text())["sample_submission"]
    val_ans = vs / "private_val/answers.csv"
    ans = pd.read_csv(val_ans, dtype=str, keep_default_na=False)
    id_col = ans.columns[0]
    sample_cols = list(pd.read_csv(vs / "public" / sample_name, nrows=0).columns)
    with tempfile.TemporaryDirectory() as t:
        perfect = Path(t) / "perfect.csv"
        _perfect(ans, sample_cols, id_col).to_csv(perfect, index=False)
        r = {"task": task.name,
             "perfect": _grade(task, val_ans, perfect, id_col),
             "val": _grade(task, val_ans, vs / "public" / sample_name, id_col),
             "test": _grade(task, task / "prepared/private/answers.csv", task / "prepared/public" / sample_name, id_col)}
    p, v, s = (r[k].get("position_score") for k in ("perfect", "val", "test"))
    problems = []
    if not r["perfect"].get("valid"):
        problems.append(f"perfect invalid: {r['perfect'].get('error')}")
    elif p is not None and p < MIN_PERFECT_POSITION:
        problems.append(f"perfect only at position {p:.2f}")
    if not r["val"].get("valid") and r["test"].get("valid"):  # (novozymes: a constant sample is NaN on both)
        problems.append(f"sample invalid on validation: {r['val'].get('error')}")
    if v is not None and s is not None and abs(v - s) > MAX_POSITION_GAP:
        problems.append(f"sample position val {v:.2f} vs test {s:.2f}")
    r["ok"] = not problems
    r["problems"] = problems
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", help="comma-separated (default: every task with a finished split)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", help="also write the full results here (JSON lines)")
    a = ap.parse_args()
    data = Path(a.data)
    tasks = [data / t for t in a.task.split(",")] if a.task else sorted(p.parent.parent for p in data.glob("*/valsplit/DONE"))
    f = open(a.json, "w") if a.json else None
    print("task\tok\tperfect_score\tperfect_pos\tval_sample_score\tval_sample_pos\ttest_sample_score\ttest_sample_pos\tproblems")
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(check, tasks):
            fmt = lambda x: "" if x is None else f"{x:.4g}"
            print("\t".join([r["task"], "ok" if r["ok"] else "CHECK",
                             *(fmt(r[k].get(m)) for k in ("perfect", "val", "test") for m in ("score", "position_score")),
                             "; ".join(r["problems"])]), flush=True)
            if f:
                f.write(json.dumps(r) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
