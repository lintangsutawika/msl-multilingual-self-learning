"""Make validation-split copies of generated MLE-Bench tasks (harbor_tasks/mlebench-low-split).

In the tasks the adapter generates, `validate` grades the full private test set: the same rows the final grade uses.
In a split copy, `validate` grades a validation set carved out of MLE-bench's *train* with the competition's own
prepare.py (valsplit.py), and the final grade is MLE-bench's, on MLE-bench's untouched test set:

  * environment/public/      replaced by the split's public data: train' (train minus the validation rows), and the
                             test inputs with the validation inputs shuffled in (both come from Kaggle's train, so
                             they can't be told apart). The agent writes one predictions.csv covering all of them.
  * environment/private_val/ the validation answers (root-only, like private/). private/ is unchanged.
  * validate.py, validate_server.py  grade only the validation rows (grade_subset.py, copied to /opt/msl).
  * tests/test.sh            grades only MLE-bench's test rows, so the score and medal are MLE-bench's.
  * instruction.md           describes the validation set.

The split must exist (valsplit.py) and should have passed the leak audit (audit_leaks.py): scripts/mlebench_tasks.sh
runs them in order. Leaderboard and score-signing key are copied unchanged, so the forwarder verifies `validate` scores of both
task sets with the same keys. Run it in the eval environment (it reads the split through MLE-bench):

    uv run --project src/benchmarks/mlebench/eval python src/benchmarks/mlebench/make_split_tasks.py \
        --tasks a,b,c [--src ...] [--dst ...]

(The earlier split, halving the test rows by id hash, is split_grade.py; it is kept for the MLE-Dojo tasks.)
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def _patch(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if old not in text:
        raise RuntimeError(f"{path}: expected text not found: {old[:60]!r}")
    path.write_text(text.replace(old, new, 1))


def _link_or_copy(src, dst):
    """Hardlink when possible (same filesystem): the split's public data is mostly MLE-bench's files."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


VALIDATE_GRADE = '''def grade(submission_path: str, work: str) -> float:
    """Score on the validation rows (grade_subset.py); MLE-bench's test answers are only used by the final grade."""
    sys.path.insert(0, "/opt/msl")
    from grade_subset import grade as grade_rows

    try:
        return grade_rows(submission_path, COMPETITION, "/workspace/private_val")
    except ValueError as e:
        sys.exit(f"validate: {e}")


def _grade_full(submission_path: str, work: str) -> float:  # the unsplit grader, unused
'''

SERVER_GRADE = '''def grade() -> float:
    import sys
    sys.path.insert(0, "/opt/msl")
    from grade_subset import grade as grade_rows
    return grade_rows(SUBMISSION, COMPETITION, "/workspace/private_val")  # validation rows only

def _grade_full() -> float:  # the unsplit grader, unused
'''

TEST_SUBSET = '''# Validation split (make_split_tasks.py): predictions.csv also covers the validation rows; grade only MLE-bench's
# test rows, so the score and medal are MLE-bench's.
if [ -f "$SUBMISSION" ]; then
    python3 -c "import sys; sys.path.insert(0, '/opt/msl'); from grade_subset import subset_csv; subset_csv(*sys.argv[1:])" \\
        "$SUBMISSION" "$COMPETITION" /workspace/private "$WORKDIR/predictions.csv"
    SUBMISSION="$WORKDIR/predictions.csv"
fi
printf '{"competition_id"'''

WORKFLOW = """# Workflow

Work iteratively: build a model, check it on the validation set with `validate <file.csv>`, improve, and resubmit. Track your score over submissions.

1. The competition data is already prepared at `/workspace/public/`: the labeled training data, the unlabeled test
   inputs you must predict, and `sample_submission.csv` (the exact submission format, one row per test id).
   Use only files in `/workspace/public/`.
2. Train an ML model on the training data to minimize the competition metric.
3. Predict **every** test id and write the predictions in the format of `sample_submission.csv`
   (same columns/order/headers). You may name intermediate submission files however you like.
4. **For every material improvement, score your file on the validation set**:
   `validate <your_submission_file.csv>` (e.g. `validate preds_v2.csv`). A fixed hidden subset of the test ids is
   the validation set; `validate` grades your predictions for those ids and prints `score=...`. **Do this after every
   material improvement**, not just at the end: the run tracks your per-submission scores and rewards a score that
   improves on your previous best.
5. When satisfied, write your **final** submission to `/workspace/predictions.csv` (this exact path is graded at the
   end, on the other test ids). Match the `sample_submission.csv` format exactly.

# Rules
- **Never read or use the hidden answers**: they are held out and strictly off-limits. Only `validate <file.csv>`
  grades against them, and it returns only a score. Reading the answers is cheating.
- Only the submitted `predictions.csv` is graded; each validate call is logged.
- Install any packages you need with `pip install --user <package>` (you run as the unprivileged user `agent`; CPU-only environment, no GPU in the default setup).
- `validate <file.csv>` (or `python validate.py <file.csv>`) is the only way to score a submission; the grader, the answers and the leaderboard are not accessible to you.
- Improve on your prior best validation score across the run.
- Keep iterating until you have a strong score; write your final `predictions.csv` once satisfied.
"""


def make(task: str, src: Path, dst: Path) -> Path:
    from src.benchmarks.mlebench.valsplit import make as valsplit

    vs = valsplit(task)  # <data>/<task>/valsplit/{public,private_val,valsplit.json}; made by scripts/mlebench_tasks.sh beforehand
    out = dst / task
    if out.exists():
        shutil.rmtree(out)
    # environment/public is replaced by the split's public data: don't copy MLE-bench's first
    shutil.copytree(src / task, out, ignore=lambda d, names: ["__pycache__"] + (["public"] if d.endswith("/environment") else []))
    env = out / "environment"
    shutil.copytree(vs / "public", env / "public", copy_function=_link_or_copy)
    shutil.copytree(vs / "private_val", env / "private_val")
    shutil.copy(vs / "valsplit.json", env / "private_val" / "valsplit.json")
    shutil.copy(HERE / "grade_subset.py", env / "grade_subset.py")
    df = env / "Dockerfile"
    _patch(df, "COPY validate.py /opt/msl/validate.py\n",
           "COPY validate.py /opt/msl/validate.py\nCOPY grade_subset.py /opt/msl/grade_subset.py\n")
    _patch(df, "COPY private/ /workspace/private/\n",
           "COPY private/ /workspace/private/\nCOPY private_val/ /workspace/private_val/\n")
    _patch(df, "chmod 600 /etc/msl/score_key /opt/msl/validate.py",
           "chmod 600 /etc/msl/score_key /opt/msl/validate.py /opt/msl/grade_subset.py")
    _patch(df, "for p in /workspace/private /workspace/leaderboard.csv",
           "for p in /workspace/private /workspace/private_val /workspace/leaderboard.csv")
    _patch(env / "validate.py", 'PROTECTED = ("/workspace/private", ',
           'PROTECTED = ("/workspace/private", "/workspace/private_val", ')
    _patch(env / "validate.py", "def grade(submission_path: str, work: str) -> float:\n", VALIDATE_GRADE)
    _patch(env / "validate_server.py", "def grade() -> float:\n", SERVER_GRADE)
    _patch(out / "tests" / "test.sh", "printf '{\"competition_id\"", TEST_SUBSET)
    inst = out / "instruction.md"
    text = inst.read_text()
    if "\n# Workflow\n" not in text:
        raise RuntimeError(f"{inst}: expected a '# Workflow' section")
    inst.write_text(text[:text.index("\n# Workflow\n") + 1] + WORKFLOW)  # keep the competition description
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--src", default=str(REPO / "harbor_tasks" / "mlebench-low"))
    ap.add_argument("--dst", default=str(REPO / "harbor_tasks" / "mlebench-low-split"))
    a = ap.parse_args()
    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for t in a.tasks.split(","):
        print(make(t, Path(a.src), dst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
