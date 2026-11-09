"""Split each task's test answers into a public and a private leaderboard, and sanity-check the grading.

Split (mirrors Kaggle): a seeded random PUBLIC_FRAC of the test *ids* is the public leaderboard (scored by
`validate` during the run); the rest is the private leaderboard (scored once, at the end). Ids are sorted before
the seeded shuffle, so the split does not depend on file order or library version. Splitting by id keeps
multi-row ids (e.g. several boxes per image) together.

Sanity: grade (a) the sample submission and (b) the true answers as a submission, on both splits. A task is usable if
- all four grade validly and (b) beats (a) on both splits in the metric's direction;
- the public split has at least `min_public_ids` ids;
- neither the sample submission nor any other public CSV reproduces the answers;
- with a Kaggle leaderboard: the leaderboard's own ranking agrees with the metric direction, (b) ranks at least
  MIN_PERFECT_POSITION and (a) at most MAX_SAMPLE_POSITION on it (otherwise our metric or test split is not
  comparable to Kaggle's, and the position reward would be meaningless or start saturated).
Results go into each config.yaml and into <root>/index.tsv.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from common import GRADER_SRC, PUBLIC_FRAC, SEED, dir_mb, read_config, task_dirs, task_overrides, write_config

INDEX_COLS = ["task", "source", "usable", "reason", "modality", "higher_is_better", "id_column", "test_ids",
              "public_ids", "private_ids", "public_mb", "public_files", "public_sample", "public_perfect",
              "private_sample", "private_perfect", "sample_position", "perfect_position", "lb_direction_ok",
              "reward", "error"]
# With a Kaggle leaderboard, the true answers must rank at least this high (fraction of teams beaten or tied).
# Lower means our metric or test split is not comparable to Kaggle's, so the position reward would be meaningless.
MIN_PERFECT_POSITION = 0.9
# ... and the sample submission must rank at most this high; above it the reward starts saturated (seen on
# shopee-product-matching 1.0, covid19-global-forecasting-week-1 0.91, yelp-recsys-2013 0.87).
MAX_SAMPLE_POSITION = 0.5


def grade(task: Path, answers: Path, submission: Path, id_col: str, leaderboard: Path | None = None) -> dict:
    cmd = [sys.executable, str(GRADER_SRC / "grade_cli.py"), "--grader-dir", str(task / "grader"),
           "--answers", str(answers), "--submission", str(submission), "--id-column", id_col]
    if leaderboard:
        cmd += ["--leaderboard", str(leaderboard)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"valid": False, "score": None, "error": f"{type(e).__name__}: {e}"[:300]}


def _informative(s: pd.Series) -> bool:
    return s.nunique(dropna=True) > 1


def _leak(ans: pd.DataFrame, other: Path, id_col: str) -> str | None:
    """Return a reason if `other` (a public CSV) holds the answers for the same rows.

    The rows are joined on the answers' id column, or on any column of `other` with exactly the answers' ids under
    another name (quora: `id` vs `test_id`). Every answer column is compared with every column of `other`, whatever
    its name (bnp: `target` vs `PredictedProb`); a match on more than 99% of rows leaks. Constant columns never count:
    an all-zero sample submission matches a 99.8%-zero target by coincidence, and key columns repeated by design
    (e.g. ImageId next to an all-zero Location) are not informative either. Answers in another layout than the
    public file (bigquery: RowId_k rows vs one column per target) are not caught here: see task_overrides.yaml."""
    try:
        o = pd.read_csv(other, low_memory=False)
    except Exception:
        return None
    if len(o) != len(ans):
        return None
    ids = ans[id_col].astype(str)
    key = id_col if id_col in o.columns else next(
        (c for c in o.columns if o[c].nunique() == len(o) and set(o[c].astype(str)) == set(ids)), None)
    if key is None:
        return None
    m = ans.merge(o.rename(columns={key: id_col}) if key != id_col else o, on=id_col, suffixes=("", "__o"))
    if len(m) != len(ans):
        return None
    def same(df, a, b):
        x, y = df[a], df[b]
        xn, yn = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
        if xn.notna().all() and yn.notna().all():
            return float(np.isclose(xn, yn, rtol=1e-9, atol=1e-12).mean())
        return float((x.astype(str) == y.astype(str)).mean())

    # Same-named columns: a leak only if ALL of them match (key columns repeated by design match on their own).
    shared = [c for c in ans.columns if c != id_col and c in o.columns and c != key]
    if shared and all(same(m, c, f"{c}__o") > 0.99 for c in shared) and any(_informative(o[c]) for c in shared):
        return f"{other.name} contains the answers (columns {shared})"
    # Answer columns the file lacks by name, against the file's columns the answers lack: the same values under
    # another name (bnp: target vs PredictedProb).
    targets = [c for c in ans.columns if c != id_col and c not in o.columns and _informative(ans[c])]
    cand = [c for c in o.columns if c != key and c not in ans.columns and _informative(o[c])]
    sample = m.sample(n=min(len(m), 2000), random_state=0)
    for t in targets:
        for c in cand:
            if same(sample, t, c) > 0.99 and same(m, t, c) > 0.99:
                return f"{other.name} contains the answers (column {c!r} = {t!r})"
    return None


def _drop_test_columns(pub_dir: Path, cols: list[str]) -> None:
    """Remove `cols` from pub_dir/test.csv (task_overrides.yaml), text otherwise unchanged; no-op once removed."""
    test = pub_dir / "test.csv"
    with open(test, newline="") as f:
        header = next(csv.reader(f))
    drop = {i for i, c in enumerate(header) if c in set(cols)}
    if not drop:
        return
    tmp = test.with_suffix(".csv.tmp")
    with open(test, newline="") as f, open(tmp, "w", newline="") as g:
        w = csv.writer(g)
        for row in csv.reader(f):
            w.writerow([v for i, v in enumerate(row) if i not in drop])
    os.replace(tmp, test)  # a new file: hardlinked copies (built tasks) keep the old one until rebuilt


MEDIA = {"image": {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".dcm", ".webp"},
         "audio": {".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg"},
         "video": {".mp4", ".avi", ".mov"}}


def modality(public: Path) -> tuple[str, int]:
    """Coarse modality of the public data (image/audio/video if any such files, else tabular/text) and file count."""
    n, found = 0, set()
    for f in public.rglob("*"):
        if f.is_file():
            n += 1
            ext = f.suffix.lower()
            found |= {k for k, v in MEDIA.items() if ext in v}
    return ("+".join(sorted(found)) or "tabular/text"), n


def public_ids(ids: list[str], frac: float, seed: int) -> set[str]:
    """Deterministic public split: shuffle the sorted unique ids with a fixed seed, take the first `frac`."""
    ids = sorted(set(ids))
    order = np.random.RandomState(seed).permutation(len(ids))
    return {ids[i] for i in order[: int(round(frac * len(ids)))]}


def lb_direction_ok(lb: Path, hib: bool) -> bool | None:
    """Does the leaderboard's own ranking agree with the metric direction? (rank 1 must be the best score)."""
    try:
        d = pd.read_csv(lb)
        s = pd.to_numeric(d["score"], errors="coerce")
        d = d.assign(score=s).dropna(subset=["score"]).sort_values("rank" if "rank" in d else d.columns[0])
    except Exception:
        return None
    if len(d) < 2 or d.score.iloc[0] == d.score.iloc[-1]:
        return None
    return bool((d.score.iloc[0] > d.score.iloc[-1]) == hib)


def split_one(task: Path, min_public_ids: int) -> dict:
    cfg = read_config(task)
    row = {"task": task.name, "source": cfg.get("source"), "usable": False, "reason": "", "error": ""}
    overrides = task_overrides()
    try:
        pub_dir, priv_dir = task / "prepared/public", task / "prepared/private"
        if task.name in overrides["drop_test_columns"]:
            _drop_test_columns(pub_dir, overrides["drop_test_columns"][task.name])
        ans = pd.read_csv(priv_dir / "answers.csv")
        sample = pub_dir / cfg["sample_submission"]
        sample_cols = list(pd.read_csv(sample, nrows=0).columns)
        id_col = sample_cols[0] if sample_cols[0] in ans.columns else ans.columns[0]
        keys = ans[id_col].astype(str)
        pub = public_ids(keys.tolist(), PUBLIC_FRAC, SEED)
        n_ids, n_pub = keys.nunique(), len(pub)
        is_pub = keys.isin(pub)
        ans[is_pub].to_csv(priv_dir / "public_answers.csv", index=False)
        ans[~is_pub].to_csv(priv_dir / "private_answers.csv", index=False)
        # The true answers in submission format (answers may carry extra columns the submission doesn't have).
        perfect = priv_dir / "_perfect_submission.csv"
        keep = [c for c in sample_cols if c in ans.columns]
        (ans[keep] if len(keep) == len(sample_cols) else ans).to_csv(perfect, index=False)
        mod, n_files = modality(pub_dir)
        row.update(id_column=id_col, test_ids=n_ids, public_ids=n_pub, private_ids=n_ids - n_pub,
                   public_mb=dir_mb(pub_dir), modality=mod, public_files=n_files)
        lb = task / cfg["leaderboard"] if cfg.get("leaderboard") else None
        hib = None
        for split in ("public", "private"):
            a = priv_dir / f"{split}_answers.csv"
            gs, gp = grade(task, a, sample, id_col, lb), grade(task, a, perfect, id_col, lb)
            row[f"{split}_sample"] = gs["score"] if gs.get("valid") else None
            row[f"{split}_perfect"] = gp["score"] if gp.get("valid") else None
            if split == "private":
                row["sample_position"] = gs.get("position_score")
                row["perfect_position"] = gp.get("position_score")
            hib = gp.get("higher_is_better", gs.get("higher_is_better"))
            row["error"] = row["error"] or (gs.get("error") or gp.get("error") or "")[:200]
        perfect.unlink()
        row["higher_is_better"] = hib
        row["lb_direction_ok"] = lb_direction_ok(lb, hib) if lb and hib is not None else None
        leak = _leak(ans, sample, id_col) or next(
            (r for r in (_leak(ans, f, id_col) for f in pub_dir.glob("*.csv") if f != sample) if r), None)

        def beats(s):
            p, q = row[f"{s}_perfect"], row[f"{s}_sample"]
            return p is not None and q is not None and (p > q if hib else p < q)

        if task.name in overrides["exclude"]:
            row["reason"] = "excluded (task_overrides.yaml): " + " ".join(str(overrides["exclude"][task.name]).split())
        elif leak:
            row["reason"] = leak
        elif None in (row["public_sample"], row["public_perfect"], row["private_sample"], row["private_perfect"]):
            row["reason"] = "grading failed"
        elif not (beats("public") and beats("private")):
            row["reason"] = "true answers do not beat the sample submission"
        elif n_pub < min_public_ids:
            row["reason"] = f"public split too small ({n_pub} < {min_public_ids} ids)"
        elif row["lb_direction_ok"] is False:
            row["reason"] = "leaderboard ranking disagrees with the metric direction"
        elif lb and row["perfect_position"] is not None and row["perfect_position"] < MIN_PERFECT_POSITION:
            row["reason"] = (f"true answers rank only {row['perfect_position']:.2f} on the leaderboard "
                             f"(< {MIN_PERFECT_POSITION}): metric or scale differs from Kaggle's")
        elif lb and row["sample_position"] is not None and row["sample_position"] > MAX_SAMPLE_POSITION:
            row["reason"] = (f"sample submission already ranks {row['sample_position']:.2f} on the leaderboard "
                             f"(> {MAX_SAMPLE_POSITION}): leaderboard not comparable, reward would start saturated")
        else:
            row["usable"] = True
        row["reward"] = cfg.get("reward")
        cfg.update(
            id_column=id_col,
            higher_is_better=hib,
            modality=mod,
            split={"seed": SEED, "public_frac": PUBLIC_FRAC, "test_ids": n_ids, "public_ids": n_pub,
                   "private_ids": n_ids - n_pub},
            sanity={k: row.get(k) for k in ("public_sample", "public_perfect", "private_sample", "private_perfect",
                                            "sample_position", "perfect_position", "lb_direction_ok")},
            usable=row["usable"],
            unusable_reason=row["reason"] or None,
            index=row,
        )
        write_config(task, cfg)
    except Exception as e:
        row.update(reason="split failed", error=f"{type(e).__name__}: {e}"[:200])
        cfg.update(usable=False, unusable_reason=row["error"], index=row)
        write_config(task, cfg)
    return row


def _split_in_subprocess(task: Path, min_public_ids: int) -> None:
    """Split one task in its own process: a crash or OOM kill fails only that task, recorded in its config."""
    r = subprocess.run([sys.executable, __file__, str(task), str(min_public_ids)], capture_output=True, text=True)
    if r.returncode != 0:
        cfg = read_config(task)
        why = "killed (likely out of memory)" if r.returncode < 0 else f"rc={r.returncode}"
        err = (r.stderr.strip().splitlines() or [""])[-1][:200]
        row = {"task": task.name, "source": cfg.get("source"), "usable": False,
               "reason": f"split crashed: {why}", "error": err}
        cfg.update(usable=False, unusable_reason=row["reason"], index=row)
        write_config(task, cfg)
    print(f"{'ok  ' if r.returncode == 0 else 'FAIL'} {task.name}", flush=True)


def run(root: Path, only: list[str] | None, workers: int, min_public_ids: int) -> None:
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda t: _split_in_subprocess(t, min_public_ids), task_dirs(root, only)))
    rows = [read_config(t).get("index") for t in task_dirs(root)]
    df = pd.DataFrame([r for r in rows if r]).reindex(columns=INDEX_COLS).sort_values(["source", "task"])
    df.to_csv(root / "index.tsv", sep="\t", index=False)
    print(df[INDEX_COLS[:10]].to_string(index=False))
    print("\nusable by source:\n" + df.assign(usable=df.usable.astype(bool)).groupby("source").usable.agg(["sum", "count"]).to_string())


if __name__ == "__main__":  # one task, used by run(): split.py <task_dir> <min_public_ids>
    split_one(Path(sys.argv[1]), int(sys.argv[2]))
