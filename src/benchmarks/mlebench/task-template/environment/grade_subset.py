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


def _ids(sub: pd.DataFrame, ans) -> tuple[str, set[str]] | None:
    """The submission's id column and the answers' ids, or None if the answers aren't keyed by it. Answers are a
    DataFrame, or a list of records (JSON-lines answers: tensorflow2-qa)."""
    col = sub.columns[0] if len(sub.columns) else None
    if isinstance(ans, pd.DataFrame) and col in ans.columns:
        return col, set(ans[col].astype(str))
    if isinstance(ans, list) and ans and isinstance(ans[0], dict) and col in ans[0]:
        return col, {str(r[col]) for r in ans}
    return None


def _subset(sub: pd.DataFrame, ans) -> pd.DataFrame:
    found = _ids(sub, ans)
    if found is None:
        return sub
    col, keep = found
    s = sub[col].astype(str)
    m = s.isin(keep)
    if not m.any():  # one answer row per several submission rows: tensorflow2-qa's <example_id>_short / _long
        m = s.str.rsplit("_", n=1).str[0].isin(keep)
        return sub[m].reset_index(drop=True)
    sub = sub[m]
    if isinstance(ans, pd.DataFrame):
        # In the answers' row order, as MLE-bench's sample was: some graders pair rows by position (uw-madison decodes
        # each predicted mask with the image size of the answer row at the same index). Key: the id, plus `class` when
        # an id has one row per class (uw-madison) or per timestamp (smartphone's trips). Rows not in the answers keep
        # their place at the end.
        key = [col] + [c for c in ("class", "UnixTimeMillis") if c in sub.columns and c in ans.columns]
        rank = {k: i for i, k in enumerate(ans[key].astype(str).itertuples(index=False, name=None))}
        order = sub[key].astype(str).apply(tuple, axis=1).map(rank)
        sub = sub.assign(_o=order.fillna(len(rank)).to_numpy()).sort_values("_o", kind="stable").drop(columns="_o")
    return sub.reset_index(drop=True)


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
