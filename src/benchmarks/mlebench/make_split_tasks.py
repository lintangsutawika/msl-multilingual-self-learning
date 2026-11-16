"""Make validate-split copies of generated MLE-Bench tasks (harbor_tasks/mlebench-low-split).

In the tasks the adapter generates, `validate` grades the full private test set: the same rows the
final grade uses. In a split copy the test rows are halved by a hash of their id
(split_grade.py, like Kaggle's public/private leaderboards): `validate` grades the val half, and the
verifier grades the final predictions.csv on both halves (val_* and heldout_* in grade.json, next
to the usual full-test score and medal). The final submission's val score is then on the scale of
the validates before it (the run 11 final-episode reward compares the two), and the heldout half is
never seen by the agent.

Data, leaderboard and score-signing key are copied unchanged, so the forwarder verifies `validate`
scores of both task sets with the same keys.

    python src/benchmarks/mlebench/make_split_tasks.py --tasks a,b,c [--src ...] [--dst ...]
"""
import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def _patch(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if old not in text:
        raise RuntimeError(f"{path}: expected text not found: {old[:60]!r}")
    path.write_text(text.replace(old, new, 1))


VALIDATE_GRADE = '''def grade(submission_path: str, work: str) -> float:
    """Score on the val half of the test rows (split_grade.py); the heldout half is only
    graded at the end of the trial."""
    ensure_data()
    sys.path.insert(0, "/opt/msl")
    from split_grade import grade_split

    r = grade_split(submission_path, "val")
    if not r["valid"]:
        raise RuntimeError("invalid submission: " + r.get("error", "the grader rejected it "
                           "(check the columns, ids and row count against sample_submission.csv)"))
    return r["score"]


def _grade_full(submission_path: str, work: str) -> float:  # the unsplit grader, unused
'''

TEST_SPLIT = '''# Split grades (val and heldout halves of the test rows, split_grade.py), merged into grade.json below.
python3 -I /opt/msl/split_grade.py "$SUBMISSION" all > "$WORKDIR/split.jsonl" 2>>"$WORKDIR/grade.log" || true
python3 - "$WORKDIR" <<'PY'
'''

TEST_MERGE = '''out.update(medal=medal, reward=tier)
for line in open(f"{work}/split.jsonl"):
    try:
        s = json.loads(line)
    except ValueError:
        continue
    for k in ("score", "valid", "medal", "above_median"):
        out[f"{s['split']}_{k}"] = s.get(k)
'''


def make(task: str, src: Path, dst: Path) -> Path:
    out = dst / task
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src / task, out, ignore=shutil.ignore_patterns("__pycache__"))
    env = out / "environment"
    (env / "split_grade.py").write_text((HERE / "split_grade.py").read_text().replace("{comp_id}", task))
    _patch(env / "validate.py", "def grade(submission_path: str, work: str) -> float:\n", VALIDATE_GRADE)
    _patch(env / "Dockerfile", "COPY validate.py /opt/msl/validate.py\n",
           "COPY validate.py /opt/msl/validate.py\nCOPY split_grade.py /opt/msl/split_grade.py\n")
    _patch(env / "Dockerfile", "chmod 600 /etc/msl/score_key /opt/msl/validate.py",
           "chmod 600 /etc/msl/score_key /opt/msl/validate.py /opt/msl/split_grade.py")
    _patch(out / "tests" / "test.sh", "python3 - \"$WORKDIR\" <<'PY'\n", TEST_SPLIT)
    _patch(out / "tests" / "test.sh", "out.update(medal=medal, reward=tier)\n", TEST_MERGE)
    inst = out / "instruction.md"
    _patch(inst, "score your submissions on the held-out test split with `validate <file.csv>`",
           "score your submissions on the validation half of the test rows with `validate <file.csv>`")
    _patch(inst, "**For every material improvement, get your score on the held-out split**",
           "**For every material improvement, get your validation score**")
    _patch(inst, "it grades that file against the held-out\n   answers and prints `score=...`.",
           "it grades that file on a fixed half of the rows of `test.csv`\n"
           "   (the validation half; rows of the other half are ignored) and prints `score=...`.")
    _patch(inst, "(this exact path is what gets graded at the end).",
           "(this exact path is what gets graded at the end, on the other half of the test rows,\n"
           "   so it must still predict every row of `test.csv`).")
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
