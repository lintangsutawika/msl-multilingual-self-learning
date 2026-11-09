"""End-to-end tests for the MLE training-data pipeline on a tiny synthetic MLE-Dojo competition (no Kaggle needed).

    pip install -r src/benchmarks/mle_train/requirements.txt pytest
    python -m pytest src/benchmarks/mle_train/tests -q
"""

import json
import re
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
    assert len(pub) + len(prv) == N_TEST and len(pub) == round(split.PUBLIC_FRAC * N_TEST)
    ids = [str(i) for i in range(5000, 5000 + N_TEST)]
    assert split.public_ids(ids, 0.5, 0) == split.public_ids(list(reversed(ids)), 0.5, 0)  # order-independent
    assert split.public_ids(ids, split.PUBLIC_FRAC, 0) == set(pub.id.astype(str))


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
    for rel in ("task.toml", "instruction.md", "environment/validate.py", "tests/test.sh", "solution/solve.sh"):
        text = (t / rel).read_text()
        for key in ("task_id", "source", "id_column", "reward_args", "description", "sample_submission", "cpus",
                    "hours", "direction", "higher_is_better"):
            assert "{" + key + "}" not in text, f"unfilled {{{key}}} in {rel}"
    assert "**lower is better**" in (t / "instruction.md").read_text()


def _load(path: Path, name: str, monkeypatch=None, env: dict | None = None):
    import importlib.util
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_validate_locked_down_and_signed(built, tmp_path, monkeypatch, capsys):
    """The box's validate prints signed scores that the rollout forwarder accepts; forged scores are rejected."""
    pytest.importorskip("websockets")
    pytest.importorskip("httpx")
    t = built["tasks"] / "toy-regression"
    env = t / "environment"
    key = env / "score_key"
    assert key.exists() and (key.stat().st_mode & 0o777) == 0o600
    assert 'user = "agent"' in (t / "task.toml").read_text()
    dockerfile = (env / "Dockerfile").read_text()
    assert "useradd" in dockerfile and "NOPASSWD: /usr/local/bin/validate-root" in dockerfile
    assert "chmod 700 /opt/grader" in dockerfile

    # Run the root-side validate.py outside the box: point its fixed paths at the built task.
    v = _load(env / "validate.py", "box_validate")
    v.GRADER, v.KEY_FILE, v.LOG = str(env / "grader"), str(key), str(tmp_path / "logs/validation_scores.jsonl")
    v.PROTECTED = (v.GRADER,)
    sample = pd.read_csv(env / "public/sample_submission.csv")

    def validate(df) -> tuple[int, str]:
        sub = tmp_path / f"sub{len(list(tmp_path.glob('sub*')))}.csv"
        df.to_csv(sub, index=False)
        monkeypatch.setattr(sys, "argv", ["validate.py", str(sub)])
        rc = v.main()
        cap = capsys.readouterr()
        return rc, cap.out.strip(), cap.err.strip()

    rc, good, _ = validate(sample)
    assert rc == 0 and re.fullmatch(r"score=\S+ sig=[0-9a-f]{16}", good)
    rc, out, err = validate(sample.head(10))  # missing ids -> invalid: no score line at all (as in MLE-bench)
    assert rc == 1 and out == "" and "invalid submission" in err
    monkeypatch.setattr(sys, "argv", ["validate.py", str(env / "grader/public_answers.csv")])
    assert v.main() == 1 and "protected" in capsys.readouterr().err
    assert len((tmp_path / "logs/validation_scores.jsonl").read_text().splitlines()) == 1  # only valid scores

    fwd = _load(PKG.parents[2] / "src/relay/node_forwarder.py", "fwd", monkeypatch,
                {"RELAY_WS_URL": "wss://x", "RELAY_SECRET": "x", "SCORE_KEYS_DIR": str(built["tasks"])})
    trial = "toy-regression__1105120000r0"
    acc, rej, signed = fwd._split_scores(trial, good)
    assert signed and rej == [] and acc == [good.split()[0][len("score="):]]
    acc, rej, _ = fwd._split_scores(trial, "score=0.0001\nscore=0.0001 sig=0123456789abcdef")
    assert acc == [] and len(rej) == 2  # the agent printing scores itself gets nothing


def test_rebuild_keeps_score_key(built):
    key = built["tasks"] / "toy-regression/environment/score_key"
    before = key.read_text()
    built["run"]("harbor", "--root", built["root"], "--out", built["tasks"])
    assert key.read_text() == before


def test_export_format_and_guard(built, tmp_path):
    rows = [json.loads(l) for l in (built["t"] / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["metadata"] == {"instance_id": "toy-regression", "source": "mledojo-dsbench"}
    assert rows[0]["prompt"].startswith("# Task")
    directions = json.loads((built["t"] / "metric_directions.json").read_text())
    assert directions == {"toy-regression": "lower"}  # the toy metric is an error (lower is better)
    assert (built["t"] / "train.dispatch.txt").read_text().strip() == "toy-regression"
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


def _run_verifier(task: Path, sub: Path | None, tmp: Path) -> Path:
    """Run the built tests/test.sh with its absolute paths (/tests, /workspace, /logs) moved under tmp."""
    for d in ("tests", "workspace", "logs/verifier"):
        (tmp / d).mkdir(parents=True, exist_ok=True)
    shutil.copytree(task / "tests", tmp / "tests", dirs_exist_ok=True)
    if sub is not None:
        shutil.copy(sub, tmp / "workspace/predictions.csv")
    script = (task / "tests/test.sh").read_text()
    for p in ("/tests/", "/workspace/", "/logs/"):
        script = script.replace(p, f"{tmp}{p}")
    script = script.replace("python3 ", f"{sys.executable} ")
    (tmp / "test.sh").write_text(script)
    r = subprocess.run(["bash", str(tmp / "test.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return tmp / "logs/verifier"


@pytest.mark.parametrize("offset,medal", [(0.1, "gold"), (0.5, "bronze"), (1.0, "none"), (None, "none")])
def test_verifier_writes_grade_json_for_trainer(built, tmp_path, offset, medal):
    """test.sh writes grade.json in the MLE-bench template's format, with MLE-bench's medal rules, and the trainer's
    final-episode reward reads it (gold 1.0 ... valid without medal 0.1, no valid submission 0)."""
    task = built["tasks"] / "toy-regression"
    sub = None
    if offset is not None:
        ans = pd.read_csv(built["root"] / "toy-regression/prepared/private/answers.csv")
        sub = tmp_path / "sub.csv"
        ans.assign(target=ans.target + offset).to_csv(sub, index=False)  # RMSE = offset
    # Toy leaderboard: 100 teams, RMSE 0.05..2.0 -> gold <= #10, silver <= #20, bronze <= #40 (Kaggle rules).
    out = _run_verifier(task, sub, tmp_path / "box")
    g = json.loads((out / "grade.json").read_text())
    assert g["medal"] == medal and g["valid_submission"] is (offset is not None) and g["is_lower_better"] in (True, None)
    assert {"score", "gold_threshold", "silver_threshold", "bronze_threshold", "median_threshold"} <= set(g)
    trial = tmp_path / "trials/toy-regression__1/toy-regression__abc"
    shutil.copytree(out, trial / "verifier")
    env = {**__import__("os").environ, "SPOOL_TRIALS_DIR": str(tmp_path / "trials"), "SPOOL_DIR": str(tmp_path / "spool")}
    code = "from src.trainer.spool_rollout import _final_reward; print(_final_reward('toy-regression__1'))"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=PKG.parents[2])
    assert r.returncode == 0, r.stderr
    want = {"gold": 1.0, "silver": 0.67, "bronze": 0.33}.get(medal, 0.1 if offset is not None else 0.0)
    assert float(r.stdout.strip().splitlines()[-1]) == want
