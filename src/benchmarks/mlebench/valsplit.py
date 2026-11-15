"""Validation split for MLE-bench tasks: carve train into train' + validation with MLE-bench's own split script.

MLE-bench made each competition's test set by splitting Kaggle's train with the competition's prepare.py (usually
`train_test_split(test_size=0.1, random_state=0)`, sometimes another ratio, an absolute size or a renumbering). We run
that same prepare.py a second time, on MLE-bench's prepared public/ as if it were Kaggle's raw data: its "train" becomes
train' and its "test" becomes our validation rows, drawn exactly the way MLE-bench drew its test set. MLE-bench's test
set and answers are untouched, so the final grade is MLE-bench's.

As in the MLE-Dojo tasks, the agent sees ONE test set and writes ONE predictions.csv: the validation inputs are
shuffled into the test inputs (both come from Kaggle's train, so they can't be told apart). `validate` scores only the
validation ids (answers in private_val/), the final grade only MLE-bench's test ids (private/).

    python -m src.benchmarks.mlebench.valsplit --competition spooky-author-identification[,...] [--force]

Writes <mlebench data>/<cid>/valsplit/{public,private_val} next to MLE-bench's prepared/ (private/ stays prepared/private).
Files are hardlinked from prepared/ where unchanged. Never writes into prepared/: prepare.py only ever sees a scratch
"raw" dir holding links to the files it reads, and every link to a file it may rewrite is a copy.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SEED = 0  # shuffles the validation rows into the test files
if not hasattr(pd.DataFrame, "applymap"):  # pandas 3 renamed it to map; MLE-bench's prepare.py (tgs-salt) asserts with it
    pd.DataFrame.applymap = pd.DataFrame.map


@dataclass
class Spec:
    # raw name prepare.py reads -> public entry it comes from (files/dirs prepare.py only reads: hardlinked/symlinked)
    raw: dict[str, str] = field(default_factory=dict)
    # raw zip prepare.py extracts -> public entries to put in it (Kaggle shipped these zipped; MLE-bench's public
    # has them unzipped). A dir entry is stored with its own name as prefix, as in Kaggle's zip.
    raw_zip: dict[str, list[str]] = field(default_factory=dict)
    # raw zip prepare.py extracts -> (public zip, folder prefix): MLE-bench re-zipped some data without the folder
    # prefix Kaggle's zip had (aerial's train.zip holds x.jpg, Kaggle's held train/x.jpg)
    raw_rezip: dict[str, tuple[str, str]] = field(default_factory=dict)
    # archive prepare.py extracts into raw/ itself -> {path under raw/: source}: the archive is shipped EMPTY and its
    # members are put in place beforehand (no multi-GB re-zip). A source is a public entry (hardlinked) or
    # "<public .7z/.zip>:<member>" (extracted from it: statoil's 7z files hold train.json, prepare.py wants
    # data/processed/train.json).
    raw_extracted: dict[str, dict[str, str]] = field(default_factory=dict)
    # name of a REMAPS function: prepare.py renumbers ids (Test_0.., 1..), so validation ids would equal MLE-bench's
    # test ids. MLE-bench's test set is left as is; the validation rows get ids after the test ids instead.
    remap: str | None = None
    # public entries that hold test inputs: MLE-bench's test rows + our validation rows. Everything else prepare.py
    # writes is train-side and REPLACES MLE-bench's version (it no longer holds the validation rows).
    test_side: list[str] = field(default_factory=list)
    # how a file is named from an answer id: None = the id itself; "image" = the part before the first "_"
    # (denoising's answers are pixels "<image>_<row>_<col>", its files <image>.png); "trip" = all but the last "-"
    # part (smartphone's answers are trips "<drive>-<phone>", its test folders <drive>/<phone>/)
    file_id: str | None = None
    # name of a PREPARES function used instead of MLE-bench's prepare.py, for competitions whose prepare.py has no
    # split to repeat (MLE-bench used Kaggle's own labelled test set)
    prepare: str | None = None
    # MLE-bench public files (globs) that are test-side but live in a directory prepare.py rewrites: linked back in
    # after the merge (siim's tfrecords/test*.tfrec)
    keep_test: list[str] = field(default_factory=list)
    # name of a CUSTOM function that builds the whole split (mlsp-2013-birds: ids are woven through a dozen files)
    custom: str | None = None
    # name of a POSTS function run on the finished public/ (siim: strip labels MLE-bench left in test files)
    post: str | None = None


# Competitions whose prepare.py keeps the original ids (no renumbering), so validation and test ids can't collide.
SPECS: dict[str, Spec] = {
    "spooky-author-identification": Spec(raw_zip={"train.zip": ["train.csv"]},
                                         test_side=["test.csv", "sample_submission.csv"]),
    "tabular-playground-series-dec-2021": Spec(raw={"train.csv": "train.csv"},
                                               test_side=["test.csv", "sample_submission.csv"]),
    "tabular-playground-series-may-2022": Spec(raw={"train.csv": "train.csv"},
                                               test_side=["test.csv", "sample_submission.csv"]),
    "new-york-city-taxi-fare-prediction": Spec(
        raw={"train.csv": "labels.csv", "GCP-Coupons-Instructions.rtf": "GCP-Coupons-Instructions.rtf"},
        test_side=["test.csv", "sample_submission.csv"]),
    "aerial-cactus-identification": Spec(raw={"train.csv": "train.csv"}, raw_rezip={"train.zip": ("train.zip", "train/")},
                                         test_side=["test.zip", "sample_submission.csv"]),
    "aptos2019-blindness-detection": Spec(raw={"train.csv": "train.csv", "train_images": "train_images"},
                                          test_side=["test.csv", "sample_submission.csv", "test_images"]),
    "dog-breed-identification": Spec(raw={"labels.csv": "labels.csv", "train": "train"},
                                     test_side=["test", "sample_submission.csv"]),
    "leaf-classification": Spec(raw_zip={"train.csv.zip": ["train.csv"], "images.zip": ["images"]},
                                test_side=["test.csv", "sample_submission.csv", "images"]),
    "random-acts-of-pizza": Spec(raw={"train.json": "train.json", "test.json": "test.json"},
                                 test_side=["test.json", "test.zip", "sampleSubmission.csv"]),
    "plant-pathology-2020-fgvc7": Spec(raw={"train.csv": "train.csv", "images": "images"}, remap="plant",
                                       test_side=["test.csv", "sample_submission.csv", "images"]),
    "nomad2018-predict-transparent-conductors": Spec(
        raw_zip={"train.zip": ["train"], "test.zip": ["test"], "train.csv.zip": ["train.csv"], "test.csv.zip": ["test.csv"]},
        remap="nomad", test_side=["test.csv", "sample_submission.csv", "test"]),
    "dogs-vs-cats-redux-kernels-edition": Spec(
        raw_rezip={"train.zip": ("train.zip", "train/"), "test.zip": ("test.zip", "test/")}, remap="dogs_vs_cats",
        test_side=["test", "test.zip", "sample_submission.csv"]),
    "text-normalization-challenge-english-language": Spec(
        raw={n: n for n in ("en_train.csv.zip", "en_test_2.csv.zip", "en_sample_submission_2.csv.zip")},
        remap="text_norm", test_side=["en_test_2.csv.zip", "en_sample_submission_2.csv.zip"]),
    "text-normalization-challenge-russian-language": Spec(
        raw={n: n for n in ("ru_train.csv.zip", "ru_test_2.csv.zip", "ru_sample_submission_2.csv.zip")},
        remap="text_norm", test_side=["ru_test_2.csv.zip", "ru_sample_submission_2.csv.zip"]),
    "denoising-dirty-documents": Spec(raw_zip={"train.zip": ["train"], "train_cleaned.zip": ["train_cleaned"]},
                                      file_id="image", test_side=["test", "sampleSubmission.csv"]),
    "histopathologic-cancer-detection": Spec(
        raw={"train_labels.csv": "train_labels.csv", "train": "train", "test": "test"},  # test/: only counted (ratio)
        test_side=["test", "sample_submission.csv"]),
    # MLE-bench holds out the records of two fixed TFRecord files (train00, train06); after its renaming those names
    # exist again, so the same rule picks two of train' as validation. Their test00/01 become test02/03.
    "siim-isic-melanoma-classification": Spec(
        raw={"train.csv": "train.csv", "train": "train", "jpeg": "jpeg", "tfrecords": "tfrecords"}, remap="siim",
        keep_test=["tfrecords/test*.tfrec"], post="siim",
        test_side=["test.csv", "sample_submission.csv", "test", "jpeg"]),
    # MLE-bench's test set is Kaggle's own (labels released after the competition): no split to repeat, so MLE-bench's
    # usual one, train_test_split(test_size=0.1, random_state=0).
    "jigsaw-toxic-comment-classification-challenge": Spec(raw={"train.csv": "train.csv"}, prepare="jigsaw",
                                                          test_side=["test.csv", "sample_submission.csv"]),
    "mlsp-2013-birds": Spec(custom="mlsp"),
    # --- medium / high: MLE-bench's prepare.py is a plain (seeded) split of train ---
    **{cid: Spec(raw={"train.csv": "train.csv"}, test_side=["test.csv", "sample_submission.csv"]) for cid in (
        "chaii-hindi-and-tamil-question-answering", "google-quest-challenge", "learning-agency-lab-automated-essay-scoring-2",
        "lmsys-chatbot-arena", "tweet-sentiment-extraction", "us-patent-phrase-to-phrase-matching")},
    "ventilator-pressure-prediction": Spec(raw={"train.csv": "train.csv"}, remap="ventilator",  # split by breath_id
                                           test_side=["test.csv", "sample_submission.csv"]),
    "petfinder-pawpularity-score": Spec(raw={"train.csv": "train.csv", "train": "train"},
                                        test_side=["test", "test.csv", "sample_submission.csv"]),
    "osic-pulmonary-fibrosis-progression": Spec(raw={"train.csv": "train.csv", "train": "train"},  # by patient
                                                test_side=["test", "test.csv", "sample_submission.csv"]),
    "predict-volcanic-eruptions-ingv-oe": Spec(raw={"train.csv": "train.csv", "train": "train"},
                                               test_side=["test", "sample_submission.csv"]),
    "AI4Code": Spec(raw={"train": "train", "train_ancestors.csv": "train_ancestors.csv",  # by notebook ancestry
                         "train_orders.csv": "train_orders.csv"}, test_side=["test", "sample_submission.csv"]),
    # champs: split by molecule; MLE-bench's public structures/ and per-atom tables cover train molecules only, so the
    # rerun drops the validation molecules from them too (train side, replaced).
    "champs-scalar-coupling": Spec(raw={n: n for n in (
        "train.csv", "structures.csv", "structures", "dipole_moments.csv", "magnetic_shielding_tensors.csv",
        "mulliken_charges.csv", "potential_energy.csv", "scalar_coupling_contributions.csv")},
        test_side=["test.csv", "sample_submission.csv"]),
    "ranzcr-clip-catheter-line-classification": Spec(
        raw={"train.csv": "train.csv", "train_annotations.csv": "train_annotations.csv", "train": "train"},
        test_side=["test", "sample_submission.csv"]),
    "kuzushiji-recognition": Spec(  # prepare.py extracts train_images.zip (flat, as MLE-bench re-zipped it)
        raw={"train.csv": "train.csv", "train_images.zip": "train_images.zip",
             "unicode_translation.csv": "unicode_translation.csv"},
        test_side=["test_images.zip", "sample_submission.csv"]),
    "tensorflow2-question-answering": Spec(raw={"simplified-nq-train.jsonl": "simplified-nq-train.jsonl"},
                                           test_side=["simplified-nq-test.jsonl", "sample_submission.csv"]),
    "smartphone-decimeter-2022": Spec(raw={"train": "train", "metadata": "metadata"},  # split by drive date
                                      test_side=["test", "sample_submission.csv"], file_id="trip"),
    "facebook-recruiting-iii-keyword-extraction": Spec(raw_extracted={"Train.zip": {"Train.csv": "train.csv"}},
                                                       test_side=["test.csv", "sample_submission.csv"]),
    "statoil-iceberg-classifier-challenge": Spec(
        raw_extracted={"train.json.7z": {"data/processed/train.json": "train.json.7z:train.json"},
                       "test.json.7z": {"data/processed/test.json": "test.json.7z:test.json"}},
        test_side=["test.json.7z", "sample_submission.csv.7z"]),
    # split by TFRecord file; the test TFRecords (renumbered) are MLE-bench's + ours, labels stripped from all
    "cassava-leaf-disease-classification": Spec(
        raw={n: n for n in ("train.csv", "train_tfrecords", "train_images", "label_num_to_disease_map.json")},
        remap="cassava", keep_test=["test_tfrecords/*.tfrec"], post="cassava",
        test_side=["test_images", "sample_submission.csv"]),
    "tensorflow-speech-recognition-challenge": Spec(raw_extracted={"train.7z": {"train/audio": "train/audio"}},
                                                    remap="speech", test_side=["test", "sample_submission.csv"]),
    "tgs-salt-identification-challenge": Spec(  # depths.csv: MLE-bench's has train ids only, so ours train' only
        raw={"depths.csv": "depths.csv"},
        raw_extracted={"competition_data.zip": {"competition_data/train.csv": "train.csv",
                                                "competition_data/train/images": "train/images",
                                                "competition_data/train/masks": "train/masks"}},
        test_side=["test", "sample_submission.csv"]),
}


# ---------------------------------------------------------------------------------------------------- file helpers
def _link_tree(src: Path, dst: Path) -> None:
    """Hardlink a file or a directory tree (same filesystem; nothing is copied)."""
    if src.is_dir():
        for p in src.rglob("*"):
            if p.is_file():
                q = dst / p.relative_to(src)
                q.parent.mkdir(parents=True, exist_ok=True)
                os.link(p, q)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)


def _read_table(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str, keep_default_na=False)  # text in, text out: no float/NA rewriting


def _merge_csv(a: Path, b: Path, out: Path) -> None:
    da, db = _read_table(a), _read_table(b)
    if list(da.columns) != list(db.columns):
        raise ValueError(f"{out.name}: columns differ between MLE-bench test and validation ({list(da.columns)[:5]} "
                         f"vs {list(db.columns)[:5]})")
    pd.concat([da, db]).sample(frac=1, random_state=SEED).to_csv(out, index=False)


def _merge_json(a: Path, b: Path, out: Path) -> None:
    la, lb = json.loads(a.read_text()), json.loads(b.read_text())
    rows = la + lb
    random.Random(SEED).shuffle(rows)
    out.write_text(json.dumps(rows, indent=4))


def _merge_zip(a: Path, b: Path, out: Path, tmp: Path) -> None:
    """Zip of one table (test.json / test.csv): merge the table. Zip of files (images): union of members."""
    with zipfile.ZipFile(a) as za, zipfile.ZipFile(b) as zb:
        na, nb = [n for n in za.namelist() if not n.endswith("/")], [n for n in zb.namelist() if not n.endswith("/")]
        if len(na) == 1 and na == nb and na[0].endswith((".csv", ".json")):
            tmp = Path(tempfile.mkdtemp(dir=tmp))  # fresh per zip: a competition can have several (text-normalization)
            (tmp / "a").mkdir()
            (tmp / "b").mkdir()
            pa, pb = Path(za.extract(na[0], tmp / "a")), Path(zb.extract(nb[0], tmp / "b"))
            merged = tmp / "merged"
            (_merge_csv if na[0].endswith(".csv") else _merge_json)(pa, pb, merged)
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zo:
                zo.write(merged, arcname=na[0])
            return
        clash = set(na) & set(nb)
        if clash:
            raise ValueError(f"{out.name}: validation files collide with test files, e.g. {sorted(clash)[:3]}")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zo:
            for z, names in ((za, na), (zb, nb)):
                for n in names:
                    zo.writestr(z.getinfo(n), z.read(n))


def _merge_lines(a: Path, b: Path, out: Path) -> None:
    """JSON lines (tensorflow2-qa's test jsonl): union of the records, shuffled, text kept as is."""
    rows = a.read_bytes().splitlines() + b.read_bytes().splitlines()
    random.Random(SEED).shuffle(rows)
    out.write_bytes(b"\n".join(rows) + b"\n")


def _merge_7z(a: Path, b: Path, out: Path, tmp: Path) -> None:
    """7z of one table (statoil's test.json.7z, sample_submission.csv.7z): merge the table."""
    import py7zr

    tmp = Path(tempfile.mkdtemp(dir=tmp))
    names = []
    for z, d in ((a, tmp / "a"), (b, tmp / "b")):
        with py7zr.SevenZipFile(z) as s:
            n = [f.filename for f in s.list() if not f.is_directory]
            s.extractall(d)
        if len(n) != 1:
            raise ValueError(f"{z.name}: expected one file inside, got {n}")
        names.append(n[0])
    if names[0] != names[1] or not names[0].endswith((".csv", ".json")):
        raise ValueError(f"{out.name}: can't merge {names}")
    merged = tmp / Path(names[0]).name
    (_merge_csv if merged.suffix == ".csv" else _merge_json)(tmp / "a" / names[0], tmp / "b" / names[0], merged)
    with py7zr.SevenZipFile(out, "w") as s:
        s.write(merged, arcname=names[0])


def _file_ids(ids: set[str], spec: Spec) -> set[str]:
    if spec.file_id == "image":
        return {i.split("_")[0] for i in ids}
    if spec.file_id == "trip":
        return {i.rsplit("-", 1)[0] for i in ids}
    return ids


def _has_id(rel: Path, ids: set[str]) -> bool:
    """Is some part of the path (a folder or file, by name or name without suffix) one of the ids?"""
    return any(q in ids or Path(q).stem in ids for q in rel.parts)


def _merge(a: Path, b: Path, out: Path, tmp: Path, test_ids: set[str]) -> None:
    """MLE-bench's test entry `a` + validation entry `b` -> `out`."""
    if a.is_dir():
        # Everything prepare.py wrote (train' files of a shared dir like images/, and the validation files), plus
        # MLE-bench's test files, picked by id (an entry's first path part, without suffix, is its id).
        _link_tree(b, out)
        for p in a.rglob("*"):
            rel = p.relative_to(a)
            if p.is_file() and _has_id(rel, test_ids):
                q = out / rel
                if q.exists():
                    raise ValueError(f"{out.name}: validation file {rel} collides with a test file")
                q.parent.mkdir(parents=True, exist_ok=True)
                os.link(p, q)
    elif a.suffix == ".csv":
        _merge_csv(a, b, out)
    elif a.suffix == ".json":
        _merge_json(a, b, out)
    elif a.suffix == ".zip":
        _merge_zip(a, b, out, tmp)
    elif a.suffix == ".jsonl":
        _merge_lines(a, b, out)
    elif a.suffix == ".7z":
        _merge_7z(a, b, out, tmp)
    else:
        raise ValueError(f"don't know how to merge {a.name}")


# ---------------------------------------------------------------------------------------------------- id remaps
def _map_csv(path: Path, col: str, f, zipped_name: str | None = None) -> None:
    """Rewrite column `col` of a CSV (or of the one CSV inside a zip) with f; text otherwise kept as read."""
    if zipped_name:
        with zipfile.ZipFile(path) as z:
            df = pd.read_csv(z.open(zipped_name), dtype=str, keep_default_na=False)
    else:
        df = _read_table(path)
    df[col] = df[col].map(f)
    tmp = path.with_name(path.name + ".tmp")
    if zipped_name:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(zipped_name, df.to_csv(index=False))
    else:
        df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _rename(paths: list[Path], f) -> None:
    """Rename files/dirs by their id (stem): two passes through temporary names, so new names never clash."""
    staged = []
    for p in paths:
        t = p.with_name(".remap-" + p.name)
        p.rename(t)
        staged.append((t, p.with_name(f(p.stem) + p.suffix)))
    for t, q in staged:
        t.rename(q)


def _remap_plant(pub, priv, ans_rel, test_ids):
    n = 1 + max(int(i.split("_")[1]) for i in test_ids)
    f = lambda v: f"Test_{int(v.split('_')[1]) + n}" if str(v).startswith("Test_") else v
    for path in (pub / "test.csv", pub / "sample_submission.csv", priv / ans_rel):
        _map_csv(path, "image_id", f)
    _rename(sorted((pub / "images").glob("Test_*.jpg")), f)


def _remap_int(n: int):
    return lambda v: str(int(v) + n)


def _remap_nomad(pub, priv, ans_rel, test_ids):
    # Kaggle numbers train and test from 1 each; start validation ids after both so every id is unique.
    train_ids = pd.read_csv(pub / "train.csv", usecols=["id"])["id"]
    f = _remap_int(max(max(int(i) for i in test_ids), int(train_ids.max())))
    for path in (pub / "test.csv", pub / "sample_submission.csv", priv / ans_rel):
        _map_csv(path, "id", f)
    _rename(sorted(d for d in (pub / "test").iterdir() if d.is_dir()), f)


def _remap_dogs_vs_cats(pub, priv, ans_rel, test_ids):
    f = _remap_int(max(int(i) for i in test_ids))
    for path in (pub / "sample_submission.csv", priv / ans_rel):
        _map_csv(path, "id", f)
    _rename(sorted((pub / "test").glob("*.jpg")), f)
    with zipfile.ZipFile(pub / "test.zip", "w", zipfile.ZIP_STORED) as z:  # as MLE-bench's compress(): flat names
        for p in sorted((pub / "test").glob("*.jpg")):
            z.write(p, arcname=p.name)


def _remap_cassava(pub, priv, ans_rel, test_ids):
    """Ids are Kaggle's image names (kept); only the test TFRecords are renumbered (ld_test00-<n>.tfrec, ...): the
    validation ones go after MLE-bench's."""
    from mlebench.registry import registry

    k = len(list(registry.get_competition("cassava-leaf-disease-classification").public_dir.glob("test_tfrecords/*")))
    _rename(sorted((pub / "test_tfrecords").glob("ld_test*.tfrec")),
            lambda s: f"ld_test{int(s[7:9]) + k:02d}{s[9:]}")


def _remap_speech(pub, priv, ans_rel, test_ids):
    """prepare.py names test clips clip_<8-digit index>.wav: the validation clips go after MLE-bench's."""
    n = 1 + max(int(i[5:13]) for i in test_ids)
    stem = lambda s: f"clip_{int(s[5:]) + n:08d}"
    for path in (pub / "sample_submission.csv", priv / ans_rel):
        _map_csv(path, "fname", lambda v: stem(v[:-4]) + ".wav")
    _rename(sorted((pub / "test" / "audio").glob("clip_*.wav")), stem)


def _remap_text_norm(pub, priv, ans_rel, test_ids):
    n = 1 + max(int(i.split("_")[0]) for i in test_ids)
    f = lambda v: f"{int(v.split('_')[0]) + n}_{v.split('_', 1)[1]}"
    lang = next(pub.glob("*_test_2.csv.zip")).name[:2]
    _map_csv(pub / f"{lang}_test_2.csv.zip", "sentence_id", lambda v: str(int(v) + n), f"{lang}_test_2.csv")
    _map_csv(pub / f"{lang}_sample_submission_2.csv.zip", "id", f, f"{lang}_sample_submission_2.csv")
    for path in [priv / ans_rel] + [p for p in priv.glob("*.csv") if p != priv / ans_rel]:
        _map_csv(path, "id", f)


def _remap_siim(pub, priv, ans_rel, test_ids):
    # The validation records come out as tfrecords/test00/01-*.tfrec, MLE-bench's test records have those names.
    n = 1 + max(int(p.name[4:6]) for p in (pub.parent.parent.parent / "prepared" / "public" / "tfrecords").glob("test*"))
    _rename(sorted((pub / "tfrecords").glob("test*.tfrec")), lambda s: f"test{int(s[4:6]) + n:02d}{s[6:]}")


def _remap_ventilator(pub, priv, ans_rel, test_ids):
    # prepare.py numbers the rows of train and test from 1 each (breath_id keeps Kaggle's values): start the
    # validation rows' ids after both, so every id is unique.
    train_ids = pd.read_csv(pub / "train.csv", usecols=["id"])["id"]
    f = _remap_int(max(max(int(i) for i in test_ids), int(train_ids.max())))
    for path in (pub / "test.csv", pub / "sample_submission.csv", priv / ans_rel):
        _map_csv(path, "id", f)


REMAPS = {"siim": _remap_siim, "plant": _remap_plant, "nomad": _remap_nomad, "dogs_vs_cats": _remap_dogs_vs_cats,
          "text_norm": _remap_text_norm, "ventilator": _remap_ventilator, "cassava": _remap_cassava,
          "speech": _remap_speech}


# ---------------------------------------------------------------------------------------------------- the split
@contextmanager
def _link_instead_of_copy():
    """While prepare.py runs, make shutil's copies hardlinks (MLE-bench's scripts copy every train image into the new
    public/; with links the validation split costs no extra disk). Falls back to a real copy if linking fails."""
    copy, copyfile, copy2, copytree = shutil.copy, shutil.copyfile, shutil.copy2, shutil.copytree

    def linker(fallback):
        def f(src, dst, *a, **k):
            d = Path(dst)
            if d.is_dir():
                d = d / Path(src).name
            try:
                if not d.exists() and Path(src).is_file():
                    os.link(os.path.realpath(src), d)
                    return str(d)
            except OSError:
                pass
            return fallback(src, dst, *a, **k)
        return f

    shutil.copy, shutil.copyfile, shutil.copy2 = linker(copy), linker(copyfile), linker(copy2)
    def linking_copytree(src, dst, *a, **k):
        if len(a) < 3:  # copy_function not passed positionally (shutil's own recursion into subdirs does)
            k["copy_function"] = shutil.copy2
        return copytree(src, dst, *a, **k)

    shutil.copytree = linking_copytree
    try:
        yield
    finally:
        shutil.copy, shutil.copyfile, shutil.copy2, shutil.copytree = copy, copyfile, copy2, copytree


# ---------------------------------------------------------------------------------------------------- own splits
JIGSAW_CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def _prepare_jigsaw(raw: Path, public: Path, private: Path) -> None:
    """MLE-bench's usual split (10%, random_state=0) of train.csv, written in the layout of its jigsaw prepare.py."""
    from sklearn.model_selection import train_test_split

    train = _read_table(raw / "train.csv")
    new_train, val = train_test_split(train, test_size=0.1, random_state=0)
    new_train.to_csv(public / "train.csv", index=False)
    val[["id", "comment_text"]].to_csv(public / "test.csv", index=False)
    sample = val[["id"]].copy()
    for c in JIGSAW_CLASSES:
        sample[c] = 0.5
    sample.to_csv(public / "sample_submission.csv", index=False)
    val[["id", *JIGSAW_CLASSES]].to_csv(private / "test.csv", index=False)


PREPARES = {"jigsaw": _prepare_jigsaw}


def _strip_tfrecord_labels(pub: Path, pattern: str = "tfrecords/test*.tfrec") -> None:
    """siim, cassava: Kaggle's train TFRecords carry each image's `target`, and MLE-bench copies some of them into
    public/ as the test records, labels included; so do the validation records. Rewrite every test record without
    `target` (a new file: the hardlinked original in prepared/ is untouched)."""
    import tensorflow as tf

    for p in sorted(pub.glob(pattern)):
        tmp = p.with_name(p.name + ".tmp")
        n = 0
        with tf.io.TFRecordWriter(str(tmp)) as w:
            for raw in tf.data.TFRecordDataset(str(p)):
                ex = tf.train.Example()
                ex.ParseFromString(raw.numpy())
                del ex.features.feature["target"]
                w.write(ex.SerializeToString())
                n += 1
        os.replace(tmp, p)
        for raw in tf.data.TFRecordDataset(str(p)).take(3):
            ex = tf.train.Example()
            ex.ParseFromString(raw.numpy())
            assert "target" not in ex.features.feature, f"leak: {p.name} still has target"
        print(f"stripped target from {p.name} ({n} records)")


POSTS = {"siim": _strip_tfrecord_labels,
         "cassava": lambda pub: _strip_tfrecord_labels(pub, "test_tfrecords/*.tfrec")}


def _make_mlsp(comp, out: Path) -> Path:
    """mlsp-2013-birds: MLE-bench's rule (20% of Kaggle's training fold, sample(frac=0.2, random_state=0)) applied to
    MLE-bench's training fold. Its prepare.py renumbers every rec_id across a dozen files, so instead of running it
    again the validation recordings keep their ids: they move to fold 1, their labels become "?", and their answers
    (Id = rec_id * 100 + species) go to private_val/answers.csv, as MLE-bench did for its test recordings."""
    pub0 = comp.public_dir
    ed = "essential_data"
    folds = pd.read_csv(pub0 / ed / "CVfolds_2.txt")
    train = folds[folds["fold"] == 0].reset_index(drop=True)
    val_recs = set(train.loc[train.sample(frac=0.2, random_state=0).index, "rec_id"].astype(int))
    test_recs = set(folds.loc[folds["fold"] == 1, "rec_id"].astype(int))
    assert not val_recs & test_recs
    n_species = len(pd.read_csv(pub0 / ed / "species_list.txt"))
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        pub, priv = Path(tmp) / "public", Path(tmp) / "private_val"
        changed = {f"{ed}/CVfolds_2.txt", f"{ed}/rec_labels_test_hidden.txt", "sample_submission.csv"}
        for p in pub0.rglob("*"):
            if p.is_file() and p.relative_to(pub0).as_posix() not in changed:
                _link_tree(p, pub / p.relative_to(pub0))
        folds.loc[folds["rec_id"].isin(val_recs), "fold"] = 1
        folds.to_csv(pub / ed / "CVfolds_2.txt", index=False)
        answers, lines = [], (pub0 / ed / "rec_labels_test_hidden.txt").read_text().splitlines()
        with open(pub / ed / "rec_labels_test_hidden.txt", "w") as f:
            f.write(lines[0] + "\n")
            for line in lines[1:]:
                rec, _, labels = line.partition(",")
                if int(rec) in val_recs:
                    assert labels != "?", f"rec {rec} has no label"
                    species = {int(s) for s in labels.split(",") if s}
                    answers += [(int(rec) * 100 + s, int(s in species)) for s in range(n_species)]
                    line = f"{rec},?"
                f.write(line + "\n")
        sample = pd.read_csv(pub0 / "sample_submission.csv")
        val_sample = pd.DataFrame({"Id": [i for i, _ in answers], "Probability": 0})
        pd.concat([sample, val_sample]).sort_values("Id").to_csv(pub / "sample_submission.csv", index=False)
        priv.mkdir()
        pd.DataFrame(answers, columns=["Id", "Probability"]).to_csv(priv / "answers.csv", index=False)
        # leak check: no validation label left in a public file
        for line in (pub / ed / "rec_labels_test_hidden.txt").read_text().splitlines()[1:]:
            rec, _, labels = line.partition(",")
            assert int(rec) not in val_recs or labels == "?", f"leak: labels of validation rec {rec}"
        shutil.move(str(pub), out / "public")
        shutil.move(str(priv), out / "private_val")
    stats = {"competition": comp.id, "id_column": "Id", "validation_ids": len(answers), "test_ids":
             len(pd.read_csv(comp.answers)), "validation_recordings": len(val_recs), "test_recordings": len(test_recs),
             "answers": "answers.csv", "seed": 0}
    (out / "valsplit.json").write_text(json.dumps(stats, indent=1) + "\n")
    (out / "DONE").touch()
    return out


CUSTOM = {"mlsp": _make_mlsp}


def _load_answers(p: Path) -> pd.DataFrame:
    """Answers as text: a CSV, or JSON lines (tensorflow2-qa; only the scalar fields are kept)."""
    if p.suffix == ".jsonl":
        rows = []
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                rows.append({k: str(v) for k, v in r.items() if not isinstance(v, (list, dict))})
        return pd.DataFrame(rows)
    return pd.read_csv(p, dtype=str)


def _empty_archive(p: Path) -> None:
    if p.suffix == ".7z":
        import py7zr

        with py7zr.SevenZipFile(p, "w"):
            pass
    else:
        with zipfile.ZipFile(p, "w"):
            pass


def _extract_member(archive: Path, member: str, dst: Path) -> None:
    if archive.suffix == ".7z":
        import py7zr

        with tempfile.TemporaryDirectory(dir=dst.parent) as t, py7zr.SevenZipFile(archive) as s:
            s.extract(t, targets=[member])
            os.replace(Path(t) / member, dst)
    else:
        with zipfile.ZipFile(archive) as z, z.open(member) as src, open(dst, "wb") as out:
            shutil.copyfileobj(src, out)


def _id_column(answers: pd.DataFrame, sample: Path) -> str:
    first = list(pd.read_csv(sample, nrows=0).columns)[0]
    return first if first in answers.columns else answers.columns[0]


def _check_no_leak(pub: Path, spec: Spec, val_ids: set[str], id_col: str) -> None:
    """No validation id may be left in a train-side file: those carry labels."""
    for p in sorted(pub.rglob("*")):
        rel = p.relative_to(pub)
        if rel.parts[0] in spec.test_side or not p.is_file():
            continue
        if p.suffix == ".csv":
            cols = list(pd.read_csv(p, nrows=0).columns)
            if id_col in cols:
                hit = val_ids & set(pd.read_csv(p, usecols=[id_col], dtype=str)[id_col])
                if hit:
                    raise AssertionError(f"leak: {rel} holds {len(hit)} validation ids, e.g. {sorted(hit)[:3]}")
        elif p.suffix == ".zip":
            with zipfile.ZipFile(p) as z:
                hit = {Path(n).stem for n in z.namelist()} & val_ids
            if hit:
                raise AssertionError(f"leak: {rel} holds validation files, e.g. {sorted(hit)[:3]}")
        elif p.suffix in (".json", ".jsonl", ".7z"):
            hit = _json_ids(p, id_col) & val_ids
            if hit:
                raise AssertionError(f"leak: {rel} holds {len(hit)} validation records, e.g. {sorted(hit)[:3]}")
        if _has_id(rel, _file_ids(val_ids, spec)):
            raise AssertionError(f"leak: train-side {rel} is (in) a validation row")


def _json_ids(p: Path, id_col: str) -> set[str]:
    """Values of the "<id_col>" key anywhere in a JSON / JSON-lines file (or the one file in a .7z), by text search:
    the files are GBs of records, and only the id matters."""
    import re

    if p.suffix == ".7z":
        import py7zr

        with tempfile.TemporaryDirectory(dir=p.parent.parent) as t, py7zr.SevenZipFile(p) as s:
            s.extractall(t)
            return set().union(*(_json_ids(q, id_col) for q in Path(t).rglob("*") if q.is_file()))
    pat = re.compile(rb'"' + re.escape(id_col.encode()) + rb'"\s*:\s*"?([^",}\s]+)')
    ids: set[str] = set()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 26), b""):  # ids can straddle chunks: keep a tail
            ids.update(m.decode() for m in pat.findall(chunk))
            if len(chunk) == 1 << 26:
                f.seek(-200, 1)
    return ids


def make(cid: str, force: bool = False) -> Path:
    from mlebench.registry import registry

    if cid not in SPECS:
        raise NotImplementedError(f"{cid}: no validation-split spec yet (see SPECS in valsplit.py)")
    spec = SPECS[cid]
    comp = registry.get_competition(cid)
    out = comp.public_dir.parent.parent / "valsplit"  # <data>/<cid>/valsplit, next to prepared/
    if (out / "DONE").exists() and not force:
        return out
    shutil.rmtree(out, ignore_errors=True)
    if spec.custom:
        out.mkdir(parents=True)
        return CUSTOM[spec.custom](comp, out)
    pub0, priv0 = comp.public_dir, comp.private_dir
    out.mkdir(parents=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        tmp = Path(tmp)
        raw, pub_v, priv_v = tmp / "raw", tmp / "public", tmp / "private"
        for d in (raw, pub_v, priv_v):
            d.mkdir()
        # The "raw" view: links to the files prepare.py reads. Directory symlinks are only used for dirs it reads
        # from (never extracts into); files are hardlinked, which prepare.py can't damage (it never opens raw for
        # writing except by extracting zips into raw/, which creates new entries).
        for name, src in spec.raw.items():
            s = pub0 / src
            if s.is_dir():
                (raw / name).symlink_to(s, target_is_directory=True)
            else:
                os.link(s, raw / name)
        for name, srcs in spec.raw_zip.items():
            with zipfile.ZipFile(raw / name, "w", zipfile.ZIP_STORED) as z:
                for src in srcs:
                    s = pub0 / src
                    for p in ([s] if s.is_file() else sorted(q for q in s.rglob("*") if q.is_file())):
                        z.write(p, arcname=p.relative_to(pub0).as_posix())
        for name, (src, prefix) in spec.raw_rezip.items():
            with zipfile.ZipFile(pub0 / src) as zi, zipfile.ZipFile(raw / name, "w", zipfile.ZIP_STORED) as zo:
                for info in zi.infolist():
                    if not info.is_dir():
                        zo.writestr(prefix + info.filename, zi.read(info))
        for name, members in spec.raw_extracted.items():
            _empty_archive(raw / name)
            for dst, src in members.items():
                d = raw / dst
                d.parent.mkdir(parents=True, exist_ok=True)
                if ":" in src:
                    _extract_member(pub0 / src.split(":")[0], src.split(":", 1)[1], d)
                elif (pub0 / src).is_dir():
                    d.symlink_to(pub0 / src, target_is_directory=True)
                else:
                    os.link(pub0 / src, d)
        with _link_instead_of_copy():
            # MLE-bench's own split, applied to MLE-bench's train
            (PREPARES[spec.prepare] if spec.prepare else comp.prepare_fn)(raw, pub_v, priv_v)

        ans_rel = comp.answers.relative_to(pub0.parent / "private")
        test_ans = _load_answers(comp.answers)
        id_col = _id_column(test_ans, comp.sample_submission)  # text-normalization keeps its sample in private/
        test_ids = set(test_ans[id_col])
        del test_ans  # tensorflow2-qa's answers hold whole documents
        if spec.remap:
            REMAPS[spec.remap](pub_v, priv_v, ans_rel, test_ids)
        val_ids = set(_load_answers(priv_v / ans_rel)[id_col])
        clash = val_ids & test_ids
        if clash:
            raise AssertionError(f"{cid}: {len(clash)} validation ids equal MLE-bench test ids (renumbered by "
                                 f"prepare.py?), e.g. {sorted(clash)[:3]}: needs an id remap before merging")

        pub = out / "public"
        pub.mkdir()
        produced = {p.name for p in pub_v.iterdir()}
        for p in pub0.iterdir():  # MLE-bench's public: test side merged, train side replaced, the rest kept
            if p.name in spec.test_side:
                if p.name not in produced:
                    raise AssertionError(f"{cid}: prepare.py wrote no validation part of {p.name}")
                _merge(p, pub_v / p.name, pub / p.name, tmp, _file_ids(test_ids, spec))
            elif p.name in produced:
                _link_tree(pub_v / p.name, pub / p.name)
            else:
                _link_tree(p, pub / p.name)  # description.md and files prepare.py doesn't touch
        for name in produced - {p.name for p in pub0.iterdir()}:
            raise AssertionError(f"{cid}: prepare.py wrote {name}, which MLE-bench's public doesn't have")
        for pattern in spec.keep_test:
            for p in pub0.glob(pattern):
                q = pub / p.relative_to(pub0)
                if q.exists():
                    raise AssertionError(f"{cid}: {q.relative_to(pub)} exists in both MLE-bench's test and validation")
                os.link(p, q)
        if spec.post:
            POSTS[spec.post](pub)
        _check_no_leak(pub, spec, val_ids, id_col)
        shutil.copytree(priv_v, out / "private_val")

    stats = {"competition": cid, "id_column": id_col, "validation_ids": len(val_ids), "test_ids": len(test_ids),
             "answers": ans_rel.as_posix(), "seed": SEED}
    (out / "valsplit.json").write_text(json.dumps(stats, indent=1) + "\n")
    (out / "DONE").touch()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--competition", required=True, help="comma-separated ids, or 'all' for every supported one")
    ap.add_argument("--force", action="store_true", help="rebuild even if done")
    a = ap.parse_args()
    ids = list(SPECS) if a.competition == "all" else [c.strip() for c in a.competition.split(",") if c.strip()]
    failed = []
    for cid in ids:
        try:
            out = make(cid, a.force)
            print(f"ok {cid}: {json.loads((out / 'valsplit.json').read_text())}")
        except Exception as e:  # keep going; report at the end
            failed.append(cid)
            print(f"FAILED {cid}: {type(e).__name__}: {e}")
    if failed:
        raise SystemExit(f"{len(failed)} failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
