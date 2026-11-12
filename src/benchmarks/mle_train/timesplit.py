"""Re-split forecasting competitions by time (task_overrides.yaml `time_split`).

MLE-Dojo/DSBench made test sets for forecasting competitions by splitting Kaggle's train at random rows, so every test
day lies inside the training period and interpolation beats real forecasting. Here MLE-Dojo's train and test (with its
answers) are merged back into Kaggle's train and the LAST stretch of time is held out as the test set, as Kaggle did.

Per task (task_overrides.yaml):
    time: column holding the time (or [year, week] for year + week number)
    holdout: share of distinct time points held out at the end (default 0.2), or
    month_days: N   hold out the last N days present in every month (bike-sharing: Kaggle's test was days 20+)
    series: columns identifying one time series (for the last-value baseline)
    id_from: how test rows get their submission id when MLE-Dojo's test.csv has none:
             "row" (1..N in test order, rossmann) or a list of columns joined by "_" (walmart: Store_Dept_Date)
    drop_test: input columns Kaggle's test never had (rossmann's Customers)

The originals are kept in prepared/_orig/ and every rerun starts from them, so the re-split is idempotent. Files are
written as new files (os.replace): the hardlinked MLE-Dojo sources are never modified.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ORIG = "_orig"


def _time(df: pd.DataFrame, spec) -> pd.Series:
    t = spec["time"]
    if isinstance(t, list):  # year + ISO week number
        y, w = df[t[0]].astype(int), df[t[1]].astype(int)
        return pd.to_datetime(y.astype(str) + "-01-01") + pd.to_timedelta(w * 7, unit="D")
    return pd.to_datetime(df[t])


def _write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def apply(task: Path, spec: dict, sample_name: str) -> dict:
    pub, priv = task / "prepared/public", task / "prepared/private"
    orig = task / "prepared" / ORIG
    if not orig.exists():  # first run: keep MLE-Dojo's files (copies, not links: the originals get replaced)
        (orig / "public").mkdir(parents=True)
        (orig / "private").mkdir()
        for f in ("train.csv", "test.csv", sample_name):
            shutil.copy2(pub / f, orig / "public" / f)
        shutil.copy2(priv / "answers.csv", orig / "private/answers.csv")
    train = pd.read_csv(orig / "public/train.csv", low_memory=False)
    test = pd.read_csv(orig / "public/test.csv", low_memory=False)
    sample = pd.read_csv(orig / "public" / sample_name)
    ans = pd.read_csv(orig / "private/answers.csv")
    id_col, targets = ans.columns[0], [c for c in ans.columns[1:]]

    # Give test rows their submission id, then their answers.
    id_from = spec.get("id_from")
    if id_from == "row":
        test[id_col] = np.arange(1, len(test) + 1)
    elif isinstance(id_from, list):
        test[id_col] = test[id_from].astype(str).agg("_".join, axis=1)
    test = test.merge(ans, on=id_col, how="inner", validate="one_to_one")
    if len(test) != len(ans):
        raise ValueError(f"only {len(test)} of {len(ans)} answers matched a test row")
    full = pd.concat([train, test], ignore_index=True)
    t = _time(full, spec)

    if "month_days" in spec:
        n = int(spec["month_days"])
        day = t.dt.normalize()
        last = day.groupby([t.dt.year, t.dt.month]).transform(lambda d: sorted(d.unique())[-n] if d.nunique() > n else d.max() + pd.Timedelta(days=1))
        is_test = day >= last
    else:
        times = np.sort(t.unique())
        k = max(1, int(round(float(spec.get("holdout", 0.2)) * len(times))))
        is_test = t >= times[-k]

    new_train = full.loc[~is_test, [c for c in train.columns]]
    test_cols = [c for c in test.columns if c not in targets and c not in spec.get("drop_test", [])]
    new_test = full.loc[is_test].copy()
    if id_from == "row":
        new_test[id_col] = np.arange(1, len(new_test) + 1)
    elif isinstance(id_from, list):
        new_test[id_col] = new_test[id_from].astype(str).agg("_".join, axis=1)
    elif new_test[id_col].isna().any():
        raise ValueError(f"held-out rows without a {id_col}")
    if new_test[id_col].duplicated().any():
        raise ValueError(f"duplicate {id_col} in the new test set")
    if id_from == "row":
        test_cols = [id_col] + [c for c in test_cols if c != id_col]
    new_ans = new_test[[id_col] + targets]
    fill = {c: sample[c].iloc[0] for c in sample.columns if c != id_col}
    new_sample = new_ans[[id_col]].assign(**fill)[list(sample.columns)]

    _write(new_train, pub / "train.csv")
    _write(new_test[test_cols], pub / "test.csv")
    _write(new_sample, pub / sample_name)
    _write(new_ans, priv / "answers.csv")
    tt = t[is_test]
    return {"train_rows": int((~is_test).sum()), "test_rows": int(is_test.sum()),
            "test_from": str(tt.min().date()), "test_to": str(tt.max().date()),
            "train_to": str(t[~is_test].max().date())}


def last_value_baseline(task: Path, spec: dict, sample_name: str, out: Path) -> None:
    """A no-skill forecast: each test row gets its series' last training value (the global last value if the series
    is new). Written in submission format to `out`."""
    pub = task / "prepared/public"
    train = pd.read_csv(pub / "train.csv", low_memory=False)
    test = pd.read_csv(pub / "test.csv", low_memory=False)
    sample = pd.read_csv(pub / sample_name)
    id_col = sample.columns[0]
    targets = [c for c in sample.columns[1:]]
    series = spec.get("series") or []
    train = train.assign(_t=_time(train, spec)).sort_values("_t")
    if series:
        last = train.groupby(series)[targets].last().reset_index()
        pred = test.merge(last, on=series, how="left")
    else:
        pred = test.assign(**{c: train[c].iloc[-1] for c in targets})
    for c in targets:
        pred[c] = pred[c].fillna(train[c].iloc[-1])
    if id_col not in pred.columns:  # rossmann/walmart test.csv: id column added by apply()
        raise ValueError(f"test.csv has no {id_col}")
    pred[[id_col] + targets].to_csv(out, index=False)
