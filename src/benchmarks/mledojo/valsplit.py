"""Validation split for the MLE-Dojo tasks: carve train into train' + validation, the way the test set was carved.

MLE-Dojo (and DSBench) made each task's test set from Kaggle's train. We carve a validation set out of the task's
train the same way, so the test set and its answers stay exactly as they are (the final grade is unchanged):

  * rule "prepare" (the MLE-Dojo Kaggle tasks): MLE-Dojo's own prepare.py for the competition is run a second time,
    on the task's prepared public/ as if it were Kaggle's raw data. Its "train" becomes train', its "test" our
    validation rows. Spec.raw / raw_zip / raw_concat map the file names prepare.py reads (Kaggle's) to ours, and
    Spec.patch removes hard-coded row counts.
  * rule "random" (DSBench: plain train.csv / test.csv): a random share of train's rows, the same share the test set
    is of train + test.
  * rule "time" (forecasting tasks, re-split by time in mle_train/timesplit.py): the last stretch of train's time
    points, the same share of time points the test period is; for bike-sharing the last days of every month.

As in the MLE-bench tasks, the agent sees ONE test set and writes ONE predictions.csv: the validation inputs are
shuffled into the test inputs. `validate` scores only the validation ids (valsplit/private_val/answers.csv), the final
grade only the test ids (prepared/private/answers.csv).

    python -m src.benchmarks.mledojo.valsplit --data DIR --task a,b,c [--force]

DIR holds one folder per task (MLE-bench layout, as mle_train/pipeline.py writes it): <task>/config.yaml,
<task>/prepared/{public,private/answers.csv}. Writes <task>/valsplit/{public,private_val/answers.csv,valsplit.json}.
Never writes into prepared/: prepare.py reads hardlinks that are made read-only while it runs.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import random
import shutil
import stat
import sys
import tempfile
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 0  # shuffles the validation rows into the test files; picks the "random" rule's rows
MLEDOJO = Path(os.environ.get("MLEDOJO_REPO", "/data/user_data/jeffc2/ext-repos/MLE-Dojo"))


@dataclass
class Spec:
    rule: str = "prepare"  # prepare | random | time
    # prepare: raw name prepare.py reads -> our public entry. Every public file is also linked under its own name.
    raw: dict[str, str] = field(default_factory=dict)
    # prepare: raw zip prepare.py extracts -> {member name: our public file}
    raw_zip: dict[str, dict[str, str]] = field(default_factory=dict)
    # prepare: raw csv -> our public csvs concatenated (kobe's data.csv: train rows + test rows without the target)
    raw_concat: dict[str, list[str]] = field(default_factory=dict)
    # prepare: replacements in prepare.py's source (asserts on Kaggle's row counts)
    patch: dict[str, str] = field(default_factory=dict)
    # public entries that hold test inputs, besides those named *test* and the sample submission
    test_side: list[str] = field(default_factory=list)
    # prepare: columns prepare.py leaves in the validation test inputs that the task's test inputs don't have (the
    # answers MLE-Dojo left in test.csv, dropped by task_overrides.yaml drop_test_columns): dropped here too
    drop_test: list[str] = field(default_factory=list)
    # train-side tables keyed by the id that carry no labels (facebook's bids.csv: bids of every bidder, as on
    # Kaggle): not leak-checked
    features: list[str] = field(default_factory=list)
    # prepare: link ONLY the Spec.raw entries into raw/ (not every public file under its own name)
    raw_only: bool = False
    # public entries left out of the split's public/: Kaggle's original download, which MLE-Dojo's prepare.py copied
    # into public/ as an "additional file". It holds Kaggle's full train, i.e. every test answer.
    drop_public: list[str] = field(default_factory=list)
    # time: {time: col | [year, week], month_days: N, id_from: "row" | [cols], drop_test: [cols]} (task_overrides.yaml)
    time: dict = field(default_factory=dict)
    note: str = ""


S = Spec
SPECS: dict[str, Spec] = {
    # --- MLE-Dojo Kaggle: prepare.py reads Kaggle's file names --------------------------------------------------
    "GiveMeSomeCredit": S(raw={"sampleEntry.csv": "sample_submission.csv"}),
    "acquire-valued-shoppers-challenge": S(features=["transactions.csv"],
                                           note="transactions.csv: purchases of every customer (test ones too, as on Kaggle), no labels"),
    "avazu-ctr-prediction": S(raw={"sampleSubmission.csv": "sample_submission.csv"}),
    "bnp-paribas-cardif-claims-management": S(drop_test=["target"]),
    "facebook-recruiting-iv-human-or-bot": S(features=["bids.csv"]),
    "loan-default-prediction": S(
        raw={"train_v2.csv": "train.csv", "test_v2.csv": "test.csv", "sampleSubmission.csv": "sample_submission.csv"},
        raw_only=True,
        drop_public=["loan-default-prediction.zip", "train_v2.csv.zip", "test_v2.csv.zip"],
        note="public/ held Kaggle's original zips (train_v2.csv.zip: every test answer): dropped"),
    "nbme-score-clinical-patient-notes": S(drop_public=["nbme-score-clinical-patient-notes.zip"],
                                           note="Kaggle's original zip in public/ holds every test answer: dropped"),
    "novozymes-enzyme-stability-prediction": S(drop_public=["novozymes-enzyme-stability-prediction.zip"],
                                               note="Kaggle's original zip in public/ holds every test answer: dropped"),
    "facial-keypoints-detection": S(
        patch={"assert total_rows == 7049": "assert True",
               "n_train_public = 5266": "n_train_public = round(total_rows * 5266 / 7049)"},
        note="prepare.py hard-codes Kaggle's 7049 rows and 5266 train rows: scaled to our train"),
    "quora-question-pairs": S(drop_test=["is_duplicate"]),
    "afsis-soil-properties": S(raw={"training.csv": "train.csv", "sorted_test.csv": "test.csv"}),
    "airbnb-recruiting-new-user-bookings": S(raw={"train_users_2.csv": "train_users.csv"}),
    "amazon-employee-access-challenge": S(raw={"sampleSubmission.csv": "sample_submission.csv"}),
    "crowdflower-weather-twitter": S(raw={"sampleSubmission.csv": "sample_submission.csv"}),
    "kobe-bryant-shot-selection": S(raw_concat={"data.csv": ["train.csv", "test.csv"]}),
    "linking-writing-processes-to-writing-quality": S(
        raw={"train_logs.csv": "train/train_logs.csv", "train_scores.csv": "train/train_scores.csv"}),
    "pubg-finish-placement-prediction": S(raw={"train_V2.csv": "train.csv", "test_V2.csv": "test.csv"}),
    "sf-crime": S(raw={"sampleSubmission.csv": "sample_submission.csv"}),
    "stumbleupon": S(raw={"train.tsv": "train_features.tsv", "raw_content": "train_raw_content"}),
    "the-winton-stock-market-challenge": S(raw_zip={"train.csv.zip": {"train.csv": "train.csv"},
                                                    "test_2.csv.zip": {"test_2.csv": "test.csv"},
                                                    "sample_submission_2.csv.zip": {"sample_submission_2.csv": "sample_submission.csv"}},
                                           patch={'assert first_window_answers == expected_first_window_ids': 'assert True'},
                                           note="prepare.py asserts the first test row is Kaggle's Id 1: removed"),
    "unimelb": S(raw={"unimelb_training.csv": "train.csv", "unimelb_example.csv": "sample_submission.csv"}),
    "word2vec-nlp-tutorial": S(
        raw={"labeledTrainData.tsv": "labeled_train.tsv", "unlabeledTrainData.tsv": "unlabeled_train.tsv"},
        patch={'assert df.shape[0] == 25000': 'assert True', 'assert train_df.shape[0] == 12500': 'assert True',
               'assert test_df.shape[0] == 12500': 'assert True'},
        note="prepare.py asserts Kaggle's row counts (25000 labeled, 12500/12500): removed"),
    "wsdm-cup-multilingual-chatbot-arena": S(),
    # --- forecasting (Kaggle): the test set is the last stretch of time (timesplit.py), so is the validation set --
    "rossmann-store-sales": S(rule="time", time={"time": "Date", "id_from": "row", "drop_test": ["Customers"]}),
    "walmart-recruiting-store-sales-forecasting": S(rule="time", time={"time": "Date", "id_from": ["Store", "Dept", "Date"]}),
    # --- forecasting (DSBench) ---------------------------------------------------------------------------------
    "bike-sharing-demand": S(rule="time", time={"time": "datetime", "month_days": 4}),
    "covid19-global-forecasting-week-2": S(rule="time", time={"time": "Date"}),
    "covid19-global-forecasting-week-3": S(rule="time", time={"time": "Date"}),
    "covid19-global-forecasting-week-4": S(rule="time", time={"time": "Date"}),
    "demand-forecasting-kernels-only": S(rule="time", time={"time": "date"}),
    "playground-series-s3e20": S(rule="time", time={"time": ["year", "week_no"]}),
    "tabular-playground-series-jan-2022": S(rule="time", time={"time": "date"}),
    "tabular-playground-series-mar-2022": S(rule="time", time={"time": "time"}),
    "tabular-playground-series-sep-2022": S(rule="time", time={"time": "date"}),
}


def spec_for(task: Path) -> Spec:
    if task.name in SPECS:
        return SPECS[task.name]
    source = _config(task).get("source", "")
    return Spec(rule="random") if source == "mledojo-dsbench" else Spec()


def _config(task: Path) -> dict:
    import yaml

    return yaml.safe_load((task / "config.yaml").read_text())


# --- files ------------------------------------------------------------------------------------------------------

def _link_tree(src: Path, dst: Path) -> None:
    if src.is_dir():
        for p in src.rglob("*"):
            if p.is_file():
                q = dst / p.relative_to(src)
                q.parent.mkdir(parents=True, exist_ok=True)
                os.link(p, q)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)


def _sep(p: Path) -> str:
    return "\t" if p.suffix == ".tsv" else ","


def _read(p: Path, **kw) -> pd.DataFrame:
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, sep=_sep(p), dtype=str, keep_default_na=False, **kw)  # text in, text out


def _write(df: pd.DataFrame, p: Path) -> None:
    if p.suffix == ".parquet":
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, sep=_sep(p), index=False)


def _merge(a: Path, b: Path, out: Path) -> None:
    """Test entry `a` + validation entry `b` -> `out` (tables: rows shuffled together; dirs: union)."""
    if a.is_dir():
        _link_tree(b, out)
        for p in a.rglob("*"):
            if not p.is_file():
                continue
            q = out / p.relative_to(a)
            if q.exists() and p.suffix in (".csv", ".tsv", ".parquet"):
                os.unlink(q)
                _merge(p, b / p.relative_to(a), q)
            elif q.exists():
                raise ValueError(f"{out.name}: validation file {q.relative_to(out)} collides with a test file")
            else:
                q.parent.mkdir(parents=True, exist_ok=True)
                os.link(p, q)
    elif a.suffix == ".json":
        la, lb = json.loads(a.read_text()), json.loads(b.read_text())
        if isinstance(la, list):
            rows = la + lb
            random.Random(SEED).shuffle(rows)
        else:  # {"images": [...], ...}: concatenate the lists that differ, keep the rest
            rows = {}
            for k, v in la.items():
                if isinstance(v, list) and v != lb.get(k):
                    v = v + lb.get(k, [])
                    random.Random(SEED).shuffle(v)
                rows[k] = v
        out.write_text(json.dumps(rows))
    elif a.suffix in (".csv", ".tsv", ".parquet"):
        da, db = _read(a), _read(b)
        if list(da.columns) != list(db.columns):
            raise ValueError(f"{out.name}: columns differ between test and validation "
                             f"({list(da.columns)[:6]} vs {list(db.columns)[:6]})")
        _write(pd.concat([da, db]).sample(frac=1, random_state=SEED), out)
    else:
        raise ValueError(f"don't know how to merge {a.name}")


@contextlib.contextmanager
def _read_only(paths: list[Path]):
    """prepare.py reads hardlinks to our files: make the files read-only meanwhile, so a write fails loudly."""
    modes = {}
    try:
        for p in paths:
            modes[p] = p.stat().st_mode
            p.chmod(modes[p] & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        yield
    finally:
        for p, m in modes.items():
            p.chmod(m)


# --- the three rules: each writes pub_v (train' + validation test side) and priv_v/answers.csv ------------------

def _run_prepare(task: Path, spec: Spec, pub0: Path, tmp: Path, pub_v: Path, priv_v: Path) -> None:
    raw = tmp / "raw"
    raw.mkdir()
    linked = []
    entries = {} if spec.raw_only else {p.name: p.name for p in pub0.iterdir()
                                        if not p.name.startswith("description") and p.name not in spec.drop_public}
    entries.update(spec.raw)
    for name, src in entries.items():
        s, d = pub0 / src, raw / name
        if s.is_dir():
            d.symlink_to(s, target_is_directory=True)
        else:
            os.link(s, d)
            linked.append(s)
    for name, members in spec.raw_zip.items():
        with zipfile.ZipFile(raw / name, "w", zipfile.ZIP_STORED) as z:
            for arc, src in members.items():
                z.write(pub0 / src, arcname=arc)
    for name, srcs in spec.raw_concat.items():
        pd.concat([_read(pub0 / s) for s in srcs]).to_csv(raw / name, index=False)
    src = (MLEDOJO / "mledojo/competitions" / task.name / "utils/prepare.py").read_text()
    for old, new in spec.patch.items():
        if old not in src:
            raise RuntimeError(f"{task.name}: prepare.py patch target not found: {old!r}")
        src = src.replace(old, new)
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("mledojo_prepare", loader=None))
    pub_v.mkdir(), priv_v.mkdir()
    random.seed(zlib.crc32(task.name.encode()))  # as mle_train/sources/mledojo.py seeded the first run
    np.random.seed(zlib.crc32(task.name.encode()))
    log = io.StringIO()
    cwd = os.getcwd()
    pd.set_option("mode.string_storage", "python")  # what the prepare scripts were written against
    try:
        os.chdir(raw)  # some scripts unzip into the working directory
        with _read_only(linked), contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            exec(compile(src, f"{task.name}/prepare.py", "exec"), mod.__dict__)
            mod.prepare(raw, pub_v, priv_v)
    except Exception as e:
        raise RuntimeError(f"{task.name}: prepare.py failed: {type(e).__name__}: {e}\n{log.getvalue()[-1500:]}") from e
    finally:
        os.chdir(cwd)
        pd.reset_option("mode.string_storage")
    for p in list(pub_v.rglob("*")) + list(priv_v.rglob("*")):  # shutil.copy2 of a raw link copies its read-only mode
        if p.is_file() and not p.is_symlink():
            p.chmod(p.stat().st_mode | stat.S_IWUSR)
    ans = priv_v / "test_answer.csv"
    if not ans.exists():
        raise RuntimeError(f"{task.name}: prepare.py wrote no test_answer.csv\n{log.getvalue()[-1500:]}")
    ans.rename(priv_v / "answers.csv")
    shutil.rmtree(raw, ignore_errors=True)


def _sample_rows(sample: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    id_col = sample.columns[0]
    fill = {c: sample[c].iloc[0] for c in sample.columns[1:]}
    return pd.DataFrame({id_col: ids.values}).assign(**fill)[list(sample.columns)]


def _run_random(task: Path, pub0: Path, ans0: pd.DataFrame, sample_name: str, pub_v: Path, priv_v: Path) -> None:
    train, test = _read(pub0 / "train.csv"), _read(pub0 / "test.csv", nrows=0)
    missing = [c for c in list(test.columns) + list(ans0.columns) if c not in train.columns]
    if missing:
        raise RuntimeError(f"{task.name}: train.csv lacks {missing}: the random rule needs a plain train/test pair")
    n = int(round(len(train) * len(ans0) / (len(train) + len(ans0))))
    rng = np.random.default_rng(zlib.crc32(task.name.encode()) + SEED)
    val = np.zeros(len(train), bool)
    val[rng.choice(len(train), n, replace=False)] = True
    pub_v.mkdir(), priv_v.mkdir()
    _write(train[~val], pub_v / "train.csv")
    _write(train[val][list(test.columns)], pub_v / "test.csv")
    _write(train[val][list(ans0.columns)], priv_v / "answers.csv")
    _write(_sample_rows(_read(pub0 / sample_name), train[val][ans0.columns[0]]), pub_v / sample_name)


def _time(df: pd.DataFrame, t) -> pd.Series:
    if isinstance(t, list):  # year + week number
        y, w = df[t[0]].astype(int), df[t[1]].astype(int)
        return pd.to_datetime(y.astype(str) + "-01-01") + pd.to_timedelta(w * 7, unit="D")
    return pd.to_datetime(df[t])


def _run_time(task: Path, spec: Spec, pub0: Path, ans0: pd.DataFrame, sample_name: str, pub_v: Path, priv_v: Path) -> None:
    ts = spec.time
    train, test = _read(pub0 / "train.csv"), _read(pub0 / "test.csv")
    id_col, targets = ans0.columns[0], list(ans0.columns[1:])
    t, t_test = _time(train, ts["time"]), _time(test, ts["time"])
    if "month_days" not in ts and t.max() >= t_test.min():  # (bike-sharing's test is days of every month)
        raise RuntimeError(f"{task.name}: train reaches {t.max()}, past the test period's start {t_test.min()}")
    if "month_days" in ts:  # the last N days present in every month (Kaggle's test was days 20+)
        n = int(ts["month_days"])
        day = t.dt.normalize()
        last = day.groupby([t.dt.year, t.dt.month]).transform(lambda d: sorted(d.unique())[-n])
        val = (day >= last).to_numpy()
    else:  # the same share of time points as the test period has
        n_tr, n_te = t.nunique(), t_test.nunique()
        k = max(1, int(round(n_tr * n_te / (n_tr + n_te))))
        val = (t >= np.sort(t.unique())[-k]).to_numpy()
    held = train[val].copy()
    id_from = ts.get("id_from")
    if id_from == "row":  # rossmann: Kaggle's ids were 1..N; validation ids come after the test ids
        held[id_col] = (np.arange(len(held)) + pd.to_numeric(ans0[id_col]).max() + 1).astype(str)
    elif isinstance(id_from, list):
        held[id_col] = held[id_from].astype(str).agg("_".join, axis=1)
    if held[id_col].duplicated().any():
        raise RuntimeError(f"{task.name}: duplicate {id_col} among the validation rows")
    pub_v.mkdir(), priv_v.mkdir()
    _write(train[~val], pub_v / "train.csv")
    _write(held[[c for c in test.columns]], pub_v / "test.csv")
    _write(held[[id_col] + targets], priv_v / "answers.csv")
    _write(_sample_rows(_read(pub0 / sample_name), held[id_col]), pub_v / sample_name)


# --- ids ----------------------------------------------------------------------------------------------------------

def _remap(pub_v: Path, priv_v: Path, id_col: str, test_ids: set[str], test_side: set[str]) -> int:
    """prepare.py renumbered the ids (0.., 1..), so validation ids equal test ids: give the validation rows ids
    after the largest test id (integers) or a "val_" prefix. Applied to every validation file holding the id column."""
    ans = _read(priv_v / "answers.csv")
    ids = ans[id_col]
    if ids.str.fullmatch(r"-?\d+").all() and all(s.lstrip("-").isdigit() for s in test_ids):
        base = max(int(s) for s in test_ids) + 1
        new = {v: str(base + i) for i, v in enumerate(ids)}
    else:
        new = {v: f"val_{v}" for v in ids}
    n = 0
    for p in [priv_v / "answers.csv"] + sorted(q for q in pub_v.rglob("*") if q.is_file()):
        rel = p.relative_to(pub_v).parts[0] if p.is_relative_to(pub_v) else None
        if rel is not None and rel not in test_side:
            continue
        if p.suffix not in (".csv", ".tsv", ".parquet"):
            continue
        df = _read(p)
        if id_col in df.columns:
            df[id_col] = df[id_col].astype(str).map(lambda v: new.get(v, v))
            _write(df, p)
            n += 1
    return n


def _json_ids(p: Path, id_col: str) -> set[str]:
    """Values of the "<id_col>" key anywhere in a JSON file (text search: only the id matters)."""
    import re

    pat = re.compile(rb'"' + re.escape(id_col.encode()) + rb'"\s*:\s*"?([^",}\s]+)')
    return {m.decode() for m in pat.findall(p.read_bytes())}


def _check_no_leak(pub: Path, test_side: set[str], val_ids: set[str], id_col: str, features=()) -> None:
    """No validation id may be left in a train-side file: those carry labels."""
    for p in sorted(pub.rglob("*")):
        rel = p.relative_to(pub)
        if rel.parts[0] in test_side or rel.parts[0] in features or not p.is_file():
            continue
        if p.suffix == ".json":
            hit = val_ids & _json_ids(p, id_col)
            if hit:
                raise AssertionError(f"leak: {rel} holds {len(hit)} validation records, e.g. {sorted(hit)[:3]}")
            continue
        if p.suffix not in (".csv", ".tsv", ".parquet"):
            continue
        cols = list(_read(p, nrows=0).columns) if p.suffix != ".parquet" else list(pd.read_parquet(p).columns)
        if id_col in cols:
            col = pd.read_parquet(p, columns=[id_col])[id_col] if p.suffix == ".parquet" else _read(p, usecols=[id_col])[id_col]
            hit = val_ids & set(col.astype(str))
            if hit:
                raise AssertionError(f"leak: {rel} holds {len(hit)} validation ids, e.g. {sorted(hit)[:3]}")


# --- main -----------------------------------------------------------------------------------------------------------

def make(task: Path, force: bool = False) -> Path:
    spec = spec_for(task)
    out = task / "valsplit"
    if (out / "DONE").exists() and not force:
        return out
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    pub0, ans_path = task / "prepared/public", task / "prepared/private/answers.csv"
    sample_name = _config(task)["sample_submission"]
    ans0 = _read(ans_path)
    id_col = ans0.columns[0]
    test_ids = set(ans0[id_col])
    test_side = {p.name for p in pub0.iterdir() if "test" in p.name.lower()} | {sample_name} | set(spec.test_side)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        tmp = Path(tmp)
        pub_v, priv_v = tmp / "public", tmp / "private"
        if spec.rule == "prepare":
            _run_prepare(task, spec, pub0, tmp, pub_v, priv_v)
        elif spec.rule == "random":
            _run_random(task, pub0, ans0, sample_name, pub_v, priv_v)
        elif spec.rule == "time":
            _run_time(task, spec, pub0, ans0, sample_name, pub_v, priv_v)
        else:
            raise ValueError(spec.rule)
        for name in test_side:
            p = pub_v / name
            if spec.drop_test and p.is_file() and p.suffix in (".csv", ".tsv"):
                df = _read(p)
                _write(df.drop(columns=[c for c in spec.drop_test if c in df.columns]), p)
        # The validation part of the sample submission: one constant row per validation id, the test sample's first
        # row's values. Never prepare.py's sample: some fill it with the true answers (ashrae, bioresponse: reset for
        # the test part by task_overrides.yaml sample_constant), some lack the target column (quora).
        _write(_sample_rows(_read(pub0 / sample_name), _read(priv_v / "answers.csv")[id_col]), pub_v / sample_name)
        val_ans = _read(priv_v / "answers.csv")
        if list(val_ans.columns) != list(ans0.columns):
            raise AssertionError(f"{task.name}: validation answers have columns {list(val_ans.columns)}, "
                                 f"the test answers {list(ans0.columns)}")
        remapped = 0
        if set(val_ans[id_col]) & test_ids:
            remapped = _remap(pub_v, priv_v, id_col, test_ids, test_side)
            val_ans = _read(priv_v / "answers.csv")
        val_ids = set(val_ans[id_col])
        clash = val_ids & test_ids
        if clash:
            raise AssertionError(f"{task.name}: {len(clash)} validation ids equal test ids, e.g. {sorted(clash)[:3]}")

        pub = out / "public"
        pub.mkdir()
        produced = {p.name for p in pub_v.iterdir()}
        for p in pub0.iterdir():  # test side merged, train side replaced, everything else kept
            if p.name in spec.drop_public:
                continue
            if p.name in test_side:
                if p.name not in produced:
                    raise AssertionError(f"{task.name}: no validation part of test-side {p.name}")
                _merge(p, pub_v / p.name, pub / p.name)
            elif p.name in produced:
                _link_tree(pub_v / p.name, pub / p.name)
            else:
                _link_tree(p, pub / p.name)  # description.md, auxiliary tables prepare.py only copied
        archives = [p.relative_to(pub).as_posix() for p in pub.rglob("*")
                    if p.is_file() and p.name.endswith((".zip", ".7z", ".rar", ".tar", ".tar.gz", ".tgz"))]
        if archives:  # MLE-Dojo's prepare.py copies Kaggle's original download into public/: it holds the answers
            raise AssertionError(f"{task.name}: archives in public/ (Kaggle's original data?): {archives}; "
                                 f"check them and list them in Spec.drop_public")
        extra = produced - {p.name for p in pub0.iterdir()} - set(spec.drop_public)
        if extra:
            raise AssertionError(f"{task.name}: the split wrote {sorted(extra)}, which the task's public/ doesn't have")
        _check_no_leak(pub, test_side, val_ids, id_col, spec.features)
        (out / "private_val").mkdir()
        shutil.copy(priv_v / "answers.csv", out / "private_val/answers.csv")

    stats = {"task": task.name, "rule": spec.rule, "id_column": id_col, "validation_ids": len(val_ids),
             "test_ids": len(test_ids), "remapped_files": remapped, "test_side": sorted(test_side), "seed": SEED}
    (out / "valsplit.json").write_text(json.dumps(stats, indent=1) + "\n")
    (out / "DONE").touch()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="folder with one MLE-bench-layout folder per task")
    ap.add_argument("--task", required=True, help="comma-separated task names")
    ap.add_argument("--force", action="store_true", help="redo splits that are already made")
    a = ap.parse_args()
    bad = 0
    for name in a.task.split(","):
        try:
            out = make(Path(a.data) / name, a.force)
            s = json.loads((out / "valsplit.json").read_text())
            print(f"ok   {name:48s} {s['rule']:8s} val={s['validation_ids']:8d} test={s['test_ids']:8d}"
                  + (f" remapped={s['remapped_files']}" if s["remapped_files"] else ""), flush=True)
        except Exception as e:
            bad += 1
            print(f"FAILED {name}: {str(e)[:600]}", flush=True)
            shutil.rmtree(Path(a.data) / name / "valsplit", ignore_errors=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
