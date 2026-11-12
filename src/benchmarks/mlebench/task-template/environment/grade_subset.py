"""Grade a submission on one subset of the test ids (validation split; see src/benchmarks/mlebench/valsplit.py).

With a validation split, /workspace/public/test.csv holds MLE-bench's test rows plus the validation rows, and the agent
predicts all of them in one file. `validate` grades the rows whose ids are in the validation answers
(/workspace/private_val), the final grade the rows whose ids are in MLE-bench's answers (/workspace/private), each with
the competition's own MLE-bench grader.

    from grade_subset import grade, subset_csv
    score = grade(submission_csv, "{comp_id}", "/workspace/private_val")   # raises with the grader's reason if invalid
    subset_csv(submission_csv, "{comp_id}", "/workspace/private", out_csv)  # keep only the MLE-bench test rows
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _answers(comp_id: str, answers_dir: str):
    from mlebench.registry import registry
    from mlebench.utils import load_answers

    comp = registry.get_competition(comp_id)
    rel = comp.answers.relative_to(comp.private_dir)
    return comp, load_answers(Path(answers_dir) / rel)


def _id_col(sub: pd.DataFrame, ans) -> str | None:
    if not isinstance(ans, pd.DataFrame):
        return None
    first = sub.columns[0] if len(sub.columns) else None
    return first if first in ans.columns else None


def _subset(sub: pd.DataFrame, ans) -> pd.DataFrame:
    col = _id_col(sub, ans)
    if col is None:
        return sub
    keep = set(ans[col].astype(str))
    return sub[sub[col].astype(str).isin(keep)].reset_index(drop=True)


def grade(submission_csv: str, comp_id: str, answers_dir: str) -> float:
    """Score of the submission's rows that belong to `answers_dir`; raises ValueError (with the grader's reason)
    if those rows are not a valid submission."""
    from mlebench.grade_helpers import InvalidSubmissionError
    from mlebench.utils import read_csv

    comp, ans = _answers(comp_id, answers_dir)
    sub = _subset(read_csv(submission_csv), ans)
    try:
        score = comp.grader.grade_fn(sub, ans)
    except InvalidSubmissionError as e:
        raise ValueError(f"invalid submission: {e}") from None
    if score is None:
        raise ValueError("invalid submission")
    return round(float(score), 5)


def subset_csv(submission_csv: str, comp_id: str, answers_dir: str, out_csv: str) -> None:
    """Write the submission's rows that belong to `answers_dir` (the final grade then runs `mlebench grade` on it)."""
    from mlebench.utils import read_csv

    _, ans = _answers(comp_id, answers_dir)
    _subset(read_csv(submission_csv), ans).to_csv(out_csv, index=False)
