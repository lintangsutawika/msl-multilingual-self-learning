"""Leak audit of MLE-bench validation splits: can the agent find any answer (MLE-bench test or our validation) in
what it sees (valsplit/public, mounted at /workspace/public)?

    python -m src.benchmarks.mlebench.audit_leaks [--competition a,b]   # default: every competition with a split

For each competition, with A = answer columns (private answers minus the id column) and ids = test + validation ids:
  * table files (csv, json, csv/json inside zips): rows whose id is in ids may not carry any column of A with
    values that agree with the answers (half or more equal, or correlation >= 0.9 for numbers; a sample
    submission's constant or random placeholder values are fine);
  * files of the train side (outside the spec's test_side entries) may not be named after a test/validation id
    (an image in train/ comes with its label in a train table);
  * TFRecords may not hold a `target` feature in test-side records.
Answer columns are what the agent must predict: the sample submission's columns (MLE-bench's answers file sometimes
holds whole test rows, inputs included). Two competitions need their own rule:
  * nomad2018: Kaggle numbers train and test from 1 each, so a train id equal to a test id is another material; train
    tables are checked against validation ids (offset past both) and by content (no test/validation material's
    inputs appear in train);
  * text-normalization: MLE-bench's sample submission copies each input token (`before`) as `after`; checked to be
    exactly that copy.
Exit code 1 if anything is found.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

from .valsplit import SPECS, Spec, _file_ids

SKIP_SUFFIXES = {".md", ".txt", ".rtf"}



def _agrees(rows: pd.DataFrame, answers: pd.DataFrame, id_col: str, col: str) -> bool:
    """Do a file's values of answer column `col` give the answers away? MLE-bench's sample submissions often hold
    random placeholders (petfinder, essay scores) or an evenly spaced sequence (google-quest): those vary but agree
    with the answers only by chance. A leak: half or more equal to the answers, or (numbers) correlation >= 0.9."""
    if col not in answers.columns:
        return True  # can't compare: keep the conservative verdict
    m = rows[[id_col, col]].astype(str).merge(answers[[id_col, col]].astype(str), on=id_col, suffixes=("_f", "_a"))
    if not len(m):
        return False
    if (m[col + "_f"] == m[col + "_a"]).mean() >= 0.5:
        return True
    f, a = pd.to_numeric(m[col + "_f"], errors="coerce"), pd.to_numeric(m[col + "_a"], errors="coerce")
    ok = f.notna() & a.notna()
    return bool(ok.sum() > 2 and f[ok].nunique() > 1 and a[ok].nunique() > 1 and f[ok].corr(a[ok]) >= 0.9)

def _tables(p: Path):
    """(name, DataFrame) for a csv/json file or each csv/json inside a zip."""
    def read(name, fh):
        if name.endswith(".csv"):
            return pd.read_csv(fh, dtype=str, keep_default_na=False)
        if name.endswith(".json"):
            data = json.load(fh)
            return pd.DataFrame(data) if isinstance(data, list) else None
        return None
    if p.suffix in (".csv", ".json"):
        with open(p, "rb") as fh:
            df = read(p.name, fh)
        if df is not None:
            yield p.name, df
    elif p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                if n.endswith((".csv", ".json")):
                    df = read(n, io.BytesIO(z.read(n)))
                    if df is not None:
                        yield f"{p.name}:{n}", df


def _check_text_norm_sample(pub: Path, sample: pd.DataFrame) -> list[str]:
    lang = next(pub.glob("*_test_2.csv.zip")).name[:2]
    (_, test), = _tables(pub / f"{lang}_test_2.csv.zip")
    before = dict(zip(test["sentence_id"] + "_" + test["token_id"], test["before"]))
    differ = (sample["after"] != sample["id"].map(before)).sum()
    return [f"{lang} sample submission: {differ} rows are not a copy of the input token"] if differ else []


def audit(cid: str) -> list[str]:
    from mlebench.registry import registry

    comp = registry.get_competition(cid)
    spec = SPECS.get(cid, Spec())
    vs = comp.public_dir.parent.parent / "valsplit"
    pub = vs / "public"
    meta = json.loads((vs / "valsplit.json").read_text())
    id_col = meta["id_column"]
    test_ans = pd.read_csv(comp.answers, dtype=str, keep_default_na=False)
    val_ans = pd.read_csv(vs / "private_val" / meta["answers"], dtype=str, keep_default_na=False)
    ids = set(test_ans[id_col]) | set(val_ans[id_col])
    samples = [q for q in pub.rglob("*") if q.is_file() and "sample" in q.name.lower()]
    sample_cols = {c for q in samples for _, df in _tables(q) for c in df.columns}
    answer_cols = [c for c in test_ans.columns if c != id_col and c in sample_cols] or \
        [c for c in test_ans.columns if c != id_col]
    file_ids = _file_ids(ids, spec)
    # Train and test numbered from 1 each by Kaggle (nomad) or by MLE-bench's prepare.py (ventilator): a train id equal
    # to a test id is another row; train-side files are checked against the validation ids (offset past both).
    native_overlap = cid in ("nomad2018-predict-transparent-conductors", "ventilator-pressure-prediction")
    val_ids = set(val_ans[id_col])
    answers = pd.concat([test_ans, val_ans]).drop_duplicates(id_col)
    problems = []
    for p in sorted(pub.rglob("*")):
        if not p.is_file() or p.suffix in SKIP_SUFFIXES:
            continue
        rel = p.relative_to(pub)
        test_side = rel.parts[0] in spec.test_side
        if p.suffix == ".tfrec":
            if p.name.startswith("test"):
                import tensorflow as tf

                for raw in tf.data.TFRecordDataset(str(p)).take(5):
                    ex = tf.train.Example()
                    ex.ParseFromString(raw.numpy())
                    if "target" in ex.features.feature:
                        problems.append(f"{rel}: test-side TFRecord holds `target`")
                        break
            continue
        if p.suffix in (".csv", ".json", ".zip"):
            for name, df in _tables(p):
                if id_col not in df.columns:
                    continue
                if "sample" in p.name.lower() and cid.startswith("text-normalization"):
                    problems += _check_text_norm_sample(pub, df)
                    continue
                check = val_ids if native_overlap and not test_side else ids
                rows = df[df[id_col].astype(str).isin(check)]
                leaked = [c for c in answer_cols if c in rows.columns and rows[c].nunique() > 1
                          and _agrees(rows, answers, id_col, c)]
                if len(rows) and leaked:
                    problems.append(f"{rel} ({name}): {len(rows)} test/validation rows carry answer columns {leaked}")
            continue
        if not test_side and (p.stem in file_ids or p.stem.split("_")[0] in file_ids and spec.file_id):
            problems.append(f"{rel}: train-side file named after a test/validation id")
    if cid == "nomad2018-predict-transparent-conductors":  # content: no test/validation material's inputs in train
        train = pd.read_csv(pub / "train.csv", dtype=str)
        test = pd.read_csv(pub / "test.csv", dtype=str)
        feats = [c for c in test.columns if c != id_col]
        # The CSV inputs are coarse (939 of MLE-bench's 2160 train rows share theirs with another row, mostly with other
        # targets): equal inputs are a leak only when the targets are equal too. Kaggle's data has a few such
        # duplicate materials (3 in MLE-bench's test, 4 in validation); they are reported, not counted as leaks.
        answers = pd.concat([test_ans, val_ans]).astype({c: float for c in feats + ["formation_energy_ev_natom",
                                                                                  "bandgap_energy_ev"]})
        tgt = ["formation_energy_ev_natom", "bandgap_energy_ev"]
        m = answers.merge(train.drop(columns=[id_col]).astype({c: float for c in feats + tgt}), on=feats + tgt)
        if m[id_col].nunique():
            print(f"       note: {m[id_col].nunique()} test/validation materials duplicate a train material "
                  f"(inputs and targets), as in Kaggle's data")
    # labels in the mlsp label file (rec_labels_test_hidden.txt): every test/validation recording must be "?"
    if cid == "mlsp-2013-birds":
        recs = {int(i) // 100 for i in ids}
        lab = pub / "essential_data" / "rec_labels_test_hidden.txt"
        for line in lab.read_text().splitlines()[1:]:
            rec, _, labels = line.partition(",")
            if int(rec) in recs and labels != "?":
                problems.append(f"{lab.relative_to(pub)}: recording {rec} (test/validation) has labels")
    return problems


def main() -> None:
    from mlebench.registry import registry

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--competition", help="comma-separated ids (default: every competition with a split)")
    a = ap.parse_args()
    data = registry.get_data_dir()
    ids = a.competition.split(",") if a.competition else sorted(
        p.parent.parent.name for p in data.glob("*/valsplit/DONE"))
    bad = 0
    for cid in ids:
        try:
            problems = audit(cid)
        except Exception as e:
            problems = [f"audit failed: {type(e).__name__}: {e}"]
        bad += bool(problems)
        print(f"{'LEAK' if problems else 'ok  '} {cid}")
        for m in problems[:10]:
            print("      ", m)
    print(f"\n{len(ids) - bad}/{len(ids)} clean")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
