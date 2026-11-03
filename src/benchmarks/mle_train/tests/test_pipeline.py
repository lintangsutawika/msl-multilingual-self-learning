"""End-to-end tests for the MLE training-data pipeline on a tiny synthetic MLE-Dojo competition (no Kaggle needed).

    pip install -r src/benchmarks/mle_train/requirements.txt pytest
    python -m pytest src/benchmarks/mle_train/tests -q
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

PKG = Path(__file__).resolve().parents[1]
PIPELINE = PKG / "pipeline.py"
GRADE_CLI = PKG / "grader/grade_cli.py"
sys.path.insert(0, str(PKG))

import split  # noqa: E402
from common import mlebench_ids  # noqa: E402

N_TEST = 1000
METRIC = '''
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError


class RmseMetrics(CompetitionMetrics):
    def __init__(self):
        super().__init__(higher_is_better=False)

    def evaluate(self, y_true, y_pred):
        y_true = y_true.sort_values("id").reset_index(drop=True)
        y_pred = y_pred.sort_values("id").reset_index(drop=True)
        return float(np.sqrt(np.mean((y_true["target"] - y_pred["target"]) ** 2)))

    def validate_submission(self, submission, ground_truth):
        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError("row count mismatch")
        if list(submission.columns) != ["id", "target"]:
            raise InvalidSubmissionError("bad columns")
        return "Submission is valid."
'''
BASE = '''
class CompetitionMetrics:
    def __init__(self, higher_is_better=True):
        self.higher_is_better = higher_is_better


class InvalidSubmissionError(Exception):
    pass
'''


def _competition(dojo: Path, data: Path, cid: str) -> None:
    rng = np.random.RandomState(1)
    (dojo / "mledojo/competitions" / cid / "utils").mkdir(parents=True)
    (dojo / "mledojo/competitions" / cid / "utils/metric.py").write_text(METRIC)
    d = data / "dsbench/prepared" / cid / "data"
    (d / "public").mkdir(parents=True)
    (d / "private").mkdir(parents=True)
    x = rng.rand(N_TEST)
    pd.DataFrame({"id": range(5000, 5000 + N_TEST), "x": x}).to_csv(d / "public/test.csv", index=False)
    pd.DataFrame({"id": range(2000), "x": rng.rand(2000), "target": rng.rand(2000)}).to_csv(d / "public/train.csv", index=False)
    pd.DataFrame({"id": range(5000, 5000 + N_TEST), "target": 0.0}).to_csv(d / "public/sample_submission.csv", index=False)
    (d / "public/description.txt").write_text(f"Predict target for {cid}.")
    pd.DataFrame({"id": range(5000, 5000 + N_TEST), "target": 3 * x}).to_csv(d / "private/test_answer.csv", index=False)
    # Kaggle-style leaderboard, lower RMSE is better: rank 1 has the smallest score.
    pd.DataFrame({"rank": range(1, 101), "score": np.linspace(0.05, 2.0, 100)}).to_csv(
        d / "private/private_leaderboard.csv", index=False)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    t = tmp_path_factory.mktemp("pipe")
    dojo, data, root, tasks = t / "MLE-Dojo", t / "data", t / "root", t / "harbor"
    (dojo / "mledojo/metrics").mkdir(parents=True)
    (dojo / "mledojo/metrics/base.py").write_text(BASE)
    _competition(dojo, data, "toy-regression")
    _competition(dojo, data, "spooky-author-identification")  # an MLE-bench id: must never get in

    def run(*args):
        r = subprocess.run([sys.executable, str(PIPELINE), *map(str, args)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r.stdout

    imp = run("mledojo-import", "--dojo-repo", dojo, "--data-dir", data, "--root", root)
    run("split", "--root", root, "--workers", "1", "--min-public-ids", "100")
    run("harbor", "--root", root, "--out", tasks)
    run("export", "--tasks-dir", tasks, "--out", t / "train.jsonl")
    return {"t": t, "root": root, "tasks": tasks, "import_out": imp, "run": run}


def grade(task_grader: Path, answers: Path, sub: Path, *extra) -> dict:
    r = subprocess.run([sys.executable, str(GRADE_CLI), "--grader-dir", str(task_grader), "--answers", str(answers),
                        "--submission", str(sub), *map(str, extra)], capture_output=True, text=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_mlebench_excluded_everywhere(built):
    assert "spooky-author-identification" in mlebench_ids()
    assert "skip spooky-author-identification: MLE-bench" in built["import_out"]
    for d in (built["root"], built["tasks"]):
        assert not (d / "spooky-author-identification").exists()
    assert "spooky" not in (built["t"] / "train.jsonl").read_text()


def test_split_disjoint_complete_deterministic(built):
    priv = built["root"] / "toy-regression/prepared/private"
    pub, prv = pd.read_csv(priv / "public_answers.csv"), pd.read_csv(priv / "private_answers.csv")
    assert set(pub.id).isdisjoint(prv.id)
    assert len(pub) + len(prv) == N_TEST and len(pub) == round(0.3 * N_TEST)
    ids = [str(i) for i in range(5000, 5000 + N_TEST)]
    assert split.public_ids(ids, 0.3, 0) == split.public_ids(list(reversed(ids)), 0.3, 0)  # order-independent
    assert split.public_ids(ids, 0.3, 0) == set(pub.id.astype(str))


def test_sanity_and_config(built):
    cfg = yaml.safe_load((built["root"] / "toy-regression/config.yaml").read_text())
    assert cfg["usable"] is True and cfg["higher_is_better"] is False and cfg["id_column"] == "id"
    s = cfg["sanity"]
    assert s["public_perfect"] == 0.0 and s["private_perfect"] == 0.0
    assert s["perfect_position"] == 1.0 and s["sample_position"] < 0.5 and s["lb_direction_ok"] is True
    assert cfg["provenance"]["imported"]
    idx = pd.read_csv(built["root"] / "index.tsv", sep="\t")
    assert idx.task.tolist() == ["toy-regression"] and bool(idx.usable.iloc[0])


def test_grade_cli_validity_and_reward(built, tmp_path):
    g = built["root"] / "toy-regression/grader"
    priv = built["root"] / "toy-regression/prepared/private"
    lb = built["root"] / "toy-regression/leaderboard.csv"
    ans = pd.read_csv(priv / "answers.csv")
    good = tmp_path / "good.csv"
    ans.assign(target=ans.target + 0.1).to_csv(good, index=False)  # all test ids, RMSE 0.1
    r = grade(g, priv / "private_answers.csv", good, "--leaderboard", lb)
    assert r["valid"] and abs(r["score"] - 0.1) < 1e-9 and r["reward"] == r["position_score"] > 0.9
    missing = tmp_path / "missing.csv"
    ans.iloc[:10].to_csv(missing, index=False)
    r = grade(g, priv / "private_answers.csv", missing, "--leaderboard", lb)
    assert not r["valid"] and r["reward"] == 0.0 and "missing" in r["error"]
    nan = tmp_path / "nan.csv"
    ans.assign(target=np.nan).to_csv(nan, index=False)
    assert grade(g, priv / "private_answers.csv", nan)["valid"] is False
    assert grade(g, priv / "private_answers.csv", tmp_path / "does-not-exist.csv")["valid"] is False
    worse, better = tmp_path / "worse.csv", tmp_path / "better.csv"
    ans.assign(target=ans.target + 1.0).to_csv(worse, index=False)
    ans.assign(target=ans.target + 0.01).to_csv(better, index=False)
    rw = grade(g, priv / "private_answers.csv", worse, "--leaderboard", lb)
    rb = grade(g, priv / "private_answers.csv", better, "--leaderboard", lb)
    assert rb["reward"] > rw["reward"]  # direction-aware: lower RMSE ranks higher


def test_harbor_task_never_ships_private_answers(built):
    t = built["tasks"] / "toy-regression"
    env_files = {p.name for p in (t / "environment").rglob("*")}
    assert "private_answers.csv" not in env_files and "answers.csv" not in env_files
    assert "leaderboard.csv" not in env_files
    assert (t / "environment/grader/public_answers.csv").exists()
    assert (t / "tests/grader/private_answers.csv").exists() and (t / "tests/grader/leaderboard.csv").exists()
    for rel in ("task.toml", "instruction.md", "environment/validate", "tests/test.sh", "solution/solve.sh"):
        text = (t / rel).read_text()
        for key in ("task_id", "source", "id_column", "reward_args", "description", "sample_submission", "cpus",
                    "hours", "direction", "higher_is_better"):
            assert "{" + key + "}" not in text, f"unfilled {{{key}}} in {rel}"
    assert "**lower is better**" in (t / "instruction.md").read_text()


def test_export_format_and_guard(built, tmp_path):
    rows = [json.loads(l) for l in (built["t"] / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["metadata"] == {"instance_id": "toy-regression", "source": "mledojo-dsbench"}
    assert rows[0]["prompt"].startswith("# Task")
    bad = tmp_path / "bad"
    shutil.copytree(built["tasks"] / "toy-regression", bad / "spooky-author-identification")
    r = subprocess.run([sys.executable, str(PIPELINE), "export", "--tasks-dir", str(bad), "--out",
                        str(tmp_path / "x.jsonl")], capture_output=True, text=True)
    assert r.returncode != 0 and "MLE-bench" in r.stderr


def test_leak_check(tmp_path):
    ans = pd.DataFrame({"id": range(1000), "y": [1] * 2 + [0] * 998})  # rare positive class
    const = tmp_path / "sample_submission.csv"
    ans.assign(y=0).to_csv(const, index=False)  # all-zero sample matches 99.8% by coincidence: not a leak
    assert split._leak(ans, const, "id") is None
    copy = tmp_path / "copy.csv"
    ans.to_csv(copy, index=False)
    assert split._leak(ans, copy, "id") == "copy.csv contains the answers"
    keys = pd.DataFrame({"id": range(1000), "img": np.arange(1000) // 30, "loc": np.random.RandomState(1).rand(1000)})
    keyed_sample = tmp_path / "keyed_sample.csv"
    keys.assign(loc=0).to_csv(keyed_sample, index=False)  # key column repeated by design, target all zero
    assert split._leak(keys, keyed_sample, "id") is None
    noisy = tmp_path / "noisy.csv"
    ans.assign(y=np.random.RandomState(0).randint(0, 2, 1000)).to_csv(noisy, index=False)
    assert split._leak(ans, noisy, "id") is None
