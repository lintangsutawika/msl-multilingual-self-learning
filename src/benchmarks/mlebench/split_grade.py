#!/usr/bin/env python3
"""Grade a submission on one half of the test rows, like Kaggle's public/private leaderboards.

The test rows are split once, by a hash of their id, into a "val" half and a "heldout" half.
`validate` grades the val half (the agent's feedback, and the training reward); the end-of-trial
verifier grades the submitted predictions.csv on both halves, so the final submission's val score
is on the same scale as the validates before it and the heldout score stays untouched by the
agent's hill-climbing. The submission still predicts every test row; rows of the other half are
dropped before grading, so the competition's own grader runs unchanged on a subset of the answers.

    python3 split_grade.py <submission.csv> [val|heldout|all]   -> one JSON line per split
"""
import hashlib
import json
import sys

COMPETITION = "{comp_id}"
SPLITS = ("val", "heldout")
SALT = "msl-split-v1"


def _group(key: str) -> str:
    """Rows that must stay together: a denoising-dirty-documents id is <image>_<row>_<col>, and
    its pixels are split by image (pixels of one page are not independent samples)."""
    if COMPETITION == "denoising-dirty-documents":
        return key.split("_", 1)[0]
    return key


def split_of(key) -> str:
    h = hashlib.sha1(f"{SALT}:{COMPETITION}:{_group(str(key))}".encode()).digest()
    return SPLITS[h[0] % 2]


def _answers():
    from mlebench.registry import registry
    from mlebench.utils import load_answers

    comp = registry.get_competition(COMPETITION)
    return comp, load_answers(comp.answers)


def grade_split(submission_path: str, split: str) -> dict:
    """{"split", "score", "valid", "above_median", "medal"} of the submission on one half (the
    medal and median against the full leaderboard: a guide, the half is a different sample)."""
    from mlebench.data import get_leaderboard
    from mlebench.utils import read_csv

    comp, answers = _answers()
    id_col = answers.columns[0]
    out = {"split": split, "score": None, "valid": False, "above_median": False, "medal": "none"}
    try:
        sub = read_csv(submission_path)
    except Exception as e:
        out["error"] = f"unreadable submission: {e}"[:300]
        return out
    keep = answers[id_col].map(split_of) == split
    ans = answers[keep].reset_index(drop=True)
    if id_col in sub.columns:
        # The whole submission must be one the full-test grader accepts (every test id exactly
        # once), else a half could pass while the final grade is invalid.
        ids, want = sub[id_col].astype(str), set(answers[id_col].astype(str))
        if len(sub) != len(answers) or ids.duplicated().any() or set(ids) != want:
            out["error"] = (f"the submission must have one row per test id: expected {len(answers)} "
                            f"rows, got {len(sub)} ({ids.nunique()} distinct ids, "
                            f"{len(want - set(ids))} test ids missing, {len(set(ids) - want)} unknown)")
            return out
        sub = sub[ids.isin(set(ans[id_col].astype(str)))].reset_index(drop=True)
    # else: leave it whole; the grader reports the missing id column as an invalid submission
    score = comp.grader(sub, ans)
    if score is None:
        return out
    rank = comp.grader.rank_score(score, get_leaderboard(comp))
    medal = ("gold" if rank["gold_medal"] else "silver" if rank["silver_medal"]
             else "bronze" if rank["bronze_medal"] else "none")
    out.update(score=float(score), valid=True, above_median=bool(rank["above_median"]), medal=medal)
    return out


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        return 2
    which = sys.argv[2] if len(sys.argv) == 3 else "all"
    for split in (SPLITS if which == "all" else (which,)):
        print(json.dumps(grade_split(sys.argv[1], split)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
