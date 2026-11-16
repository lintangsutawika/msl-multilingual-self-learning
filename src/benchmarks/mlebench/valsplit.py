"""Validation split for MLE-bench tasks: carve train into train' + validation with MLE-bench's own split script.

MLE-bench made each competition's test set by splitting Kaggle's train with the competition's prepare.py (usually
`train_test_split(test_size=0.1, random_state=0)`, sometimes another ratio, an absolute size or a renumbering). We run
that same prepare.py a second time, on MLE-bench's prepared public/ as if it were Kaggle's raw data: its "train" becomes
train' and its "test" becomes our validation rows, drawn exactly the way MLE-bench drew its test set. MLE-bench's test
set and answers are untouched, so the final grade is MLE-bench's.

As in the MLE-Dojo tasks, the agent sees ONE test set and writes ONE predictions.csv: the validation inputs are
shuffled into the test inputs (both come from Kaggle's train, so they can't be told apart). `validate` scores only the
validation ids (answers in private_val/), the final grade only MLE-bench's test ids (private/).

    $GRADER_PY -m src.benchmarks.mlebench.valsplit --competition spooky-author-identification[,...] [--force]

Run it with the grader venv (setup_grader_venv.sh: pandas<3): MLE-bench's prepare.py scripts break under pandas 3.

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
    # part (smartphone's answers are trips "<drive>-<phone>", its test folders <drive>/<phone>/); "seqpos" = all but the
    # last "_" part (stanford-covid's answers are "<id>_<position>"); "caseday" = the first two "_" parts (uw-madison's
    # answers are "<case>_<day>_slice_<n>", its folders <case>/<case>_<day>/); "nfl" = the game play's three videos
    # (answers "<game>_<play>_<step>_<p1>_<p2>", videos "<game>_<play>_{All29,Endzone,Sideline}.mp4")
    file_id: str | None = None
    # name of a PREPARES function used instead of MLE-bench's prepare.py, for competitions whose prepare.py has no
    # split to repeat (MLE-bench used Kaggle's own labelled test set)
    prepare: str | None = None
    # MLE-bench public files (globs) that are test-side but live in a directory prepare.py rewrites: linked back in
    # after the merge (siim's tfrecords/test*.tfrec)
    keep_test: list[str] = field(default_factory=list)
    # key of the record id in JSON / JSON-lines files, when it isn't the answers' id column (stanford-covid: records
    # "id", answers "<id>_<position>"): the leak check looks for validation records by it (through file_id)
    json_id: str | None = None
    # public entries prepare.py rewrites but MLE-bench's version is kept (hubmap's dataset information lists every
    # slide, MLE-bench's test slides included; prepare.py would keep only train')
    keep_public: list[str] = field(default_factory=list)
    # name of a CUSTOM function that builds the whole split (mlsp-2013-birds: ids are woven through a dozen files)
    custom: str | None = None
    # name of a POSTS function run on the finished public/ (siim: strip labels MLE-bench left in test files)
    post: str | None = None
    # test-side dirs that hold ONLY test files in MLE-bench's public (hms' test_spectrograms/, named by a second id):
    # merged whole instead of picking MLE-bench's files by answer id
    whole_test: list[str] = field(default_factory=list)


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
    "bms-molecular-translation": Spec(  # images nested by id: train/a/b/c/abc....png
        raw={n: n for n in ("train_labels.csv", "train", "extra_approved_InChIs.csv")},
        test_side=["test", "sample_submission.csv"]),
    "rsna-miccai-brain-tumor-radiogenomic-classification": Spec(  # one folder of scans per patient id
        raw={"train_labels.csv": "train_labels.csv", "train": "train"}, test_side=["test", "sample_submission.csv"]),
    "hotel-id-2021-fgvc8": Spec(raw={"train.csv": "train.csv", "train_images": "train_images"},
                                test_side=["test_images", "sample_submission.csv"]),
    "plant-pathology-2021-fgvc8": Spec(raw={"train.csv": "train.csv", "train_images": "train_images"},
                                       test_side=["test_images", "sample_submission.csv"]),
    "alaska2-image-steganalysis": Spec(raw={d: d for d in ("Cover", "JMiPOD", "JUNIWARD", "UERD")},
                                       remap="alaska2", test_side=["Test", "sample_submission.csv"]),
    "hubmap-kidney-segmentation": Spec(
        raw={n: n for n in ("train.csv", "HuBMAP-20-dataset_information.csv", "train")}, prepare="hubmap",
        keep_public=["HuBMAP-20-dataset_information.csv"], post="hubmap", test_side=["test", "sample_submission.csv"]),
    # one image of each whale seen twice goes to test (unseen test whales become new_whale), as MLE-bench did; a lone
    # image of a whale MLE-bench's test asks for stays in train' (prepare="whale")
    "whale-categorization-playground": Spec(raw={"train.csv": "train.csv", "train": "train"}, prepare="whale",
                                            test_side=["test", "sample_submission.csv"]),
    # MLE-bench graded Kaggle's released test labels (no split to repeat): own 10% splits of train (prepare=...)
    "jigsaw-unintended-bias-in-toxicity-classification": Spec(raw={"train.csv": "train.csv"}, prepare="jigsaw_ub",
                                                              test_side=["test.csv", "sample_submission.csv"]),
    "freesound-audio-tagging-2019": Spec(
        raw={n: n for n in ("train_curated.csv", "train_curated.zip", "sample_submission.csv")}, prepare="freesound",
        test_side=["test.zip", "sample_submission.csv"]),
    # test_v2.txt.zip is written whole by the prepare (MLE-bench's test rows + validation rows), so not test_side
    "billion-word-imputation": Spec(raw={n: n for n in ("train_v2.txt.zip", "test_v2.txt.zip")}, prepare="billion"),
    # time split: the week before MLE-bench's test week is validation (customer ids get "_val", remap="hm")
    "h-and-m-personalized-fashion-recommendations": Spec(
        raw={n: n for n in ("transactions_train.csv", "articles.csv", "customers.csv", "images",
                            "sample_submission.csv")}, remap="hm", test_side=["sample_submission.csv"]),
    # MLE-bench's test = Kaggle's training4 archive; validation = training3 (prepare="gesture")
    "multi-modal-gesture-recognition": Spec(
        raw={"training.csv": "training.csv", "training3.tar.gz": "training3.tar.gz"}, prepare="gesture",
        post="gesture", test_side=["test.csv", "randomPredictions.csv", "test.tar.gz"]),
    "nfl-player-contact-detection": Spec(  # by game play, as MLE-bench
        raw={n: n for n in ("train_labels.csv", "train_baseline_helmets.csv", "train_player_tracking.csv",
                            "train_video_metadata.csv", "train")}, file_id="nfl",
        test_side=["test", "test_baseline_helmets.csv", "test_player_tracking.csv", "test_video_metadata.csv",
                   "sample_submission.csv"]),
    "uw-madison-gi-tract-image-segmentation": Spec(  # by case, as MLE-bench
        raw={"train.csv": "train.csv", "train": "train"}, file_id="caseday",
        test_side=["test", "test.csv", "sample_submission.csv"]),
    "imet-2020-fgvc7": Spec(raw={n: n for n in ("train.csv", "labels.csv", "train")},
                            test_side=["test", "sample_submission.csv"]),
    "cdiscount-image-classification-challenge": Spec(  # prepare.py re-filters train.bson (~56 GB) three times
        raw={n: n for n in ("train.bson", "train_example.bson", "category_names.csv")},
        test_side=["test.bson", "sample_submission.csv"]),
    "herbarium-2020-fgvc7": Spec(custom="herbarium",  # test images renumbered: see _make_herbarium
                                 test_side=["nybg2020/test", "sample_submission.csv"]),  # (for the audit)
    "hms-harmful-brain-activity-classification": Spec(  # by spectrogram_id, as MLE-bench
        raw={n: n for n in ("train.csv", "example_figures", "train_eegs", "train_spectrograms")},
        test_side=["test.csv", "sample_submission.csv", "test_eegs", "test_spectrograms"],
        whole_test=["test_eegs", "test_spectrograms"]),
    # MLE-bench's test = train fragment 3 ("a"); validation = fragment 2 ("b"), train' = fragment 1
    "vesuvius-challenge-ink-detection": Spec(raw={"train": "train"}, prepare="vesuvius",
                                             test_side=["test", "sample_submission.csv"]),
    "iwildcam-2019-fgvc6": Spec(  # by camera location, as MLE-bench (prepare.py extracts train_images.zip itself)
        raw={"train.csv": "train.csv", "train_images.zip": "train_images.zip"},
        test_side=["test.csv", "test_images.zip", "sample_submission.csv"]),
    "siim-covid19-detection": Spec(  # by study, as MLE-bench; answers "<study>_study" / "<image>_image",
        raw={n: n for n in ("train_study_level.csv", "train_image_level.csv", "train")},  # files test/<study>/../<image>.dcm
        file_id="seqpos", test_side=["test", "sample_submission.csv"]),
    "iwildcam-2020-fgvc7": Spec(  # by camera location, as MLE-bench
        raw={n: n for n in ("iwildcam2020_train_annotations.json", "iwildcam2020_test_information.json",
                            "iwildcam2020_megadetector_results.json", "train")},
        test_side=["test", "iwildcam2020_test_information.json", "sample_submission.csv"]),
    "stanford-covid-vaccine": Spec(  # test.json is JSON lines; prepare.py reads its columns from raw test.json
        raw={n: n for n in ("train.json", "test.json", "sample_submission.csv")}, remap="covid",
        file_id="seqpos", json_id="id", test_side=["test.json", "sample_submission.csv"]),
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
    if isinstance(la, dict):  # COCO-style metadata (iwildcam-2020's test_information): concatenate the lists that
        rows = {}             # differ (images), keep the rest (categories, info) as MLE-bench's
        for k, v in la.items():
            if isinstance(v, list) and v != lb.get(k):
                v = v + lb.get(k, [])
                random.Random(SEED).shuffle(v)
            rows[k] = v
        out.write_text(json.dumps(rows))
        return
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
    if spec.file_id == "seqpos":
        return {i.rsplit("_", 1)[0] for i in ids}
    if spec.file_id == "caseday":
        return {"_".join(i.split("_")[:2]) for i in ids}
    if spec.file_id == "nfl":
        return {f"{'_'.join(i.split('_')[:2])}_{v}" for i in ids for v in ("All29", "Endzone", "Sideline")}
    return ids


def _has_id(rel: Path, ids: set[str]) -> bool:
    """Is some part of the path (a folder or file, by name or name without suffix) one of the ids?"""
    return any(q in ids or Path(q).stem in ids or q.split(".")[0] in ids or q.split("-")[0] in ids for q in rel.parts)


def _merge(a: Path, b: Path, out: Path, tmp: Path, test_ids: set[str], whole: bool = False) -> None:
    """MLE-bench's test entry `a` + validation entry `b` -> `out`."""
    if a.is_dir() and whole:
        _link_tree(b, out)
        for p in a.rglob("*"):
            if p.is_file():
                q = out / p.relative_to(a)
                if q.exists():
                    raise ValueError(f"{out.name}: validation file {q.relative_to(out)} collides with a test file")
                q.parent.mkdir(parents=True, exist_ok=True)
                os.link(p, q)
    elif a.is_dir():
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
        try:
            _merge_json(a, b, out)
        except json.JSONDecodeError:  # JSON lines named .json (stanford-covid)
            _merge_lines(a, b, out)
    elif a.suffix == ".zip":
        _merge_zip(a, b, out, tmp)
    elif a.suffix == ".jsonl":
        _merge_lines(a, b, out)
    elif a.suffix == ".7z":
        _merge_7z(a, b, out, tmp)
    elif a.name.endswith(".tar.gz"):
        _merge_tar(a, b, out)
    elif a.suffix == ".bson":  # a stream of documents (cdiscount's test.bson): concatenated
        with open(out, "wb") as fo:
            for src in (a, b):
                with open(src, "rb") as fi:
                    shutil.copyfileobj(fi, fo, 64 << 20)
    else:
        raise ValueError(f"don't know how to merge {a.name}")


def _merge_tar(a: Path, b: Path, out: Path) -> None:
    """tar.gz of files (multi-modal-gesture's Sample*.zip): union of members, light compression."""
    import tarfile

    seen = set()
    with tarfile.open(out, "w:gz", compresslevel=1) as to:
        for src in (a, b):
            with tarfile.open(src) as ti:
                for m in ti:
                    if m.isdir():
                        continue
                    if m.name in seen:
                        raise ValueError(f"{out.name}: validation file {m.name} collides with a test file")
                    seen.add(m.name)
                    to.addfile(m, ti.extractfile(m))


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


def _remap_covid(pub, priv, ans_rel, test_ids):
    """Ids are Kaggle's (kept); prepare.py numbers the test records' `index` 0..n-1: the validation records go after
    MLE-bench's."""
    from mlebench.registry import registry

    m = sum(1 for _ in open(registry.get_competition("stanford-covid-vaccine").public_dir / "test.json"))
    rows = [json.loads(line) for line in (pub / "test.json").read_text().splitlines()]
    for r in rows:
        r["index"] += m
    (pub / "test.json").write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))


def _remap_alaska2(pub, priv, ans_rel, test_ids):
    """prepare.py names test images 0001.jpg..5000.jpg (4 digits; train images 00001.jpg.. have 5): the validation
    images go after MLE-bench's, 5001..9999, and the last one is 0000 (unused: prepare.py counts from 1), so every
    test name keeps 4 digits and none equals a train name (10000.jpg would)."""
    n = max(int(i[:-4]) for i in test_ids)
    stem = lambda s: f"{int(s) + n if int(s) + n < 10000 else 0:04d}"
    for path in (pub / "sample_submission.csv", priv / ans_rel):
        _map_csv(path, "Id", lambda v: stem(v[:-4]) + ".jpg")
    _rename(sorted((pub / "Test").glob("*.jpg")), stem)


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


def _remap_hm(pub, priv, ans_rel, test_ids):
    """h-and-m: validation and test answers are both keyed by customer_id (different weeks): validation rows become
    <customer_id>_val."""
    f = lambda v: f"{v}_val"
    _map_csv(priv / ans_rel, "customer_id", f)
    _map_csv(pub / "sample_submission.csv", "customer_id", f)


REMAPS = {"hm": _remap_hm, "siim": _remap_siim, "plant": _remap_plant, "nomad": _remap_nomad, "dogs_vs_cats": _remap_dogs_vs_cats,
          "text_norm": _remap_text_norm, "ventilator": _remap_ventilator, "cassava": _remap_cassava,
          "speech": _remap_speech, "covid": _remap_covid, "alaska2": _remap_alaska2}


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
                if d.exists() and os.path.samefile(src, d):  # copied twice (hotel-id lists some images twice)
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


def _prepare_hubmap(raw: Path, public: Path, private: Path) -> None:
    """MLE-bench split Kaggle's 15 train slides 12 + 3 (train_test_split(train_size=12, test_size=3)): repeated on its
    12 as 9 + 3, the same 3 held out."""
    import sys

    from mlebench.registry import registry
    from sklearn.model_selection import train_test_split

    fn = registry.get_competition("hubmap-kidney-segmentation").prepare_fn
    mod = sys.modules[fn.__module__]
    mod.train_test_split = lambda *a, **k: train_test_split(*a, **{**k, "train_size": 9})
    try:
        fn(raw, public, private)
    finally:
        mod.train_test_split = train_test_split


def _prepare_whale(raw: Path, public: Path, private: Path) -> None:
    """MLE-bench's whale split, except that a whale with a single train image that MLE-bench's test asks for keeps it
    in train': its prepare.py sends a lone image to validation 10% of the time, which would leave that test whale
    unlearnable. Those images sit out MLE-bench's split and are added to train' afterwards."""
    from mlebench.registry import registry

    comp = registry.get_competition("whale-categorization-playground")
    train = _read_table(raw / "train.csv")
    asked = set(_read_table(comp.answers)["Id"]) - {"new_whale"}
    n = train["Id"].value_counts()
    keep = train["Id"].isin([w for w in asked if n.get(w, 0) == 1])
    raw2 = raw.parent / "raw_whale"
    raw2.mkdir()
    train[~keep].to_csv(raw2 / "train.csv", index=False)
    (raw2 / "train").symlink_to((raw / "train").resolve(), target_is_directory=True)
    comp.prepare_fn(raw2, public, private)
    pd.concat([_read_table(public / "train.csv"), train[keep]]).to_csv(public / "train.csv", index=False)
    for img in train.loc[keep, "Image"]:
        os.link((raw / "train" / img).resolve(), public / "train" / img)
    print(f"whale: {int(keep.sum())} lone train images of MLE-bench test whales kept in train'")


def _prepare_jigsaw_ub(raw: Path, public: Path, private: Path) -> None:
    """jigsaw-unintended: MLE-bench used Kaggle's released test labels; here its usual split (10%, random_state=0) of
    train.csv. Answers in the layout of Kaggle's test_private_expanded.csv (train's `target` is its `toxicity`)."""
    from sklearn.model_selection import train_test_split

    train = _read_table(raw / "train.csv")
    new_train, val = train_test_split(train, test_size=0.1, random_state=0)
    new_train.to_csv(public / "train.csv", index=False)
    val[["id", "comment_text"]].to_csv(public / "test.csv", index=False)
    val[["id"]].assign(prediction=0.0).to_csv(public / "sample_submission.csv", index=False)
    val.rename(columns={"target": "toxicity"}).to_csv(private / "test.csv", index=False)


def _prepare_freesound(raw: Path, public: Path, private: Path) -> None:
    """freesound-2019: MLE-bench used Kaggle's released test labels; here 10% (random_state=0) of the curated train
    clips (the test clips are curated-quality too) become validation: train_curated.zip/.csv lose them, test.zip gets
    them. train_noisy is untouched. Answers many-hot over the classes, as MLE-bench's."""
    from sklearn.model_selection import train_test_split

    cur = _read_table(raw / "train_curated.csv")
    new_cur, val = train_test_split(cur, test_size=0.1, random_state=0)
    new_cur.to_csv(public / "train_curated.csv", index=False)
    held = set(val["fname"])
    with zipfile.ZipFile(raw / "train_curated.zip") as zi, \
            zipfile.ZipFile(public / "train_curated.zip", "w", zipfile.ZIP_STORED) as zt, \
            zipfile.ZipFile(public / "test.zip", "w", zipfile.ZIP_STORED) as zv:
        for info in zi.infolist():
            if not info.is_dir():
                (zv if info.filename in held else zt).writestr(info, zi.read(info))
    classes = list(pd.read_csv(raw / "sample_submission.csv", nrows=0).columns[1:])
    hot = pd.DataFrame([[f] + [int(c in set(l.split(","))) for c in classes] for f, l in zip(val["fname"], val["labels"])],
                       columns=["fname", *classes])
    hot.to_csv(private / "test.csv", index=False)
    hot.assign(**{c: 0 for c in classes}).to_csv(public / "sample_submission.csv", index=False)  # as MLE-bench's


def _prepare_billion(raw: Path, public: Path, private: Path) -> None:
    """billion-word: MLE-bench's prepare.py run on its own train, except (1) the line count is counted, not Kaggle's
    hardcoded 30301028 (its final assert would fail), and (2) validation ids follow MLE-bench's test ids (its ids
    start at 0 again). Writes test_v2.txt.zip with MLE-bench's test rows + the validation rows (not merged later: it
    is a .txt table), private/test.csv with the validation answers only."""
    import numpy as np

    rng = np.random.RandomState(0)
    with zipfile.ZipFile(raw / "test_v2.txt.zip") as z:
        test_txt = z.read("test_v2.txt").decode()
    n_test = test_txt.count("\n") - 1  # header line
    tmp = Path(tempfile.mkdtemp(dir=public.parent))
    val_lines = []
    with zipfile.ZipFile(raw / "train_v2.txt.zip") as z, z.open("train_v2.txt") as f, \
            open(tmp / "train_v2.txt", "w") as tr, open(private / "test.csv", "w") as pv:
        pv.write('"id","sentence"\n')
        i = n_test
        for raw_line in f:
            sentence = raw_line.decode()
            if rng.uniform() <= 0.01 and len(sentence.strip().split()) > 2:
                s = sentence.strip().replace('"', '""')
                words = s.split()
                k = rng.randint(1, len(words) - 1)
                pv.write(f'{i},"{s}"\n')
                val_lines.append(f'{i},"{" ".join(words[:k] + words[k + 1:])}"\n')
                i += 1
            else:
                tr.write(sentence)
    with zipfile.ZipFile(public / "train_v2.txt.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(tmp / "train_v2.txt", arcname="train_v2.txt")
    rows = test_txt.splitlines(keepends=True)
    with zipfile.ZipFile(public / "test_v2.txt.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("test_v2.txt", "".join(rows[:1] + rows[1:] + val_lines))
    shutil.rmtree(tmp)
    print(f"billion-word: {len(val_lines)} validation sentences, ids {n_test}..{n_test + len(val_lines) - 1}")


def _prepare_gesture(raw: Path, public: Path, private: Path) -> None:
    """multi-modal-gesture: MLE-bench held out the last training archive (training4) as test; repeated on its three:
    training3's samples are validation (test.tar.gz, test.csv, randomPredictions.csv, answers), as its prepare.py
    writes them. training3.tar.gz itself is removed from public/ (post="gesture")."""
    import random
    import tarfile

    with tarfile.open(raw / "training3.tar.gz") as t:
        val_ids = sorted(Path(m.name).stem[-4:] for m in t if m.name.endswith(".zip"))
    tr = pd.read_csv(raw / "training.csv", dtype={"Id": str, "Sequence": str})
    tr[~tr["Id"].isin(val_ids)].to_csv(public / "training.csv", index=False)
    ans = tr[tr["Id"].isin(val_ids)]
    assert len(ans) == len(val_ids), f"{len(ans)} labels for {len(val_ids)} training3 samples"
    ans.to_csv(private / "test.csv", index=False)
    pd.DataFrame({"Id": val_ids}).to_csv(public / "test.csv", index=False)
    random.seed(0)
    preds = [" ".join(str(x) for x in random.sample(range(1, 21), 20)) for _ in val_ids]
    pd.DataFrame({"Id": val_ids, "Sequence": preds}).to_csv(public / "randomPredictions.csv", index=False)
    os.link(raw / "training3.tar.gz", public / "test.tar.gz")


def _prepare_vesuvius(raw: Path, public: Path, private: Path) -> None:
    """vesuvius: MLE-bench held out the last of Kaggle's 3 train fragments (3) as test fragment "a"; repeated on its
    two: fragment 2 is validation fragment "b", written as its prepare.py writes a test fragment (no ink labels, no
    ir.png; answers = inklabels_rle.csv with the ir.png size)."""
    from PIL import Image

    frag = raw / "train" / "2"
    _link_tree(raw / "train" / "1", public / "train" / "1")
    rle = _read_table(frag / "inklabels_rle.csv")
    assert len(rle) == 1, f"fragment 2: {len(rle)} RLE rows"
    with Image.open(frag / "ir.png") as img:
        width, height = img.size
    rle = rle.assign(width=width, height=height, Id="b")
    rle.to_csv(private / "inklabels_rle.csv", index=False)
    for f in frag.rglob("*"):
        if f.is_file() and f.name not in ("inklabels.png", "inklabels_rle.csv", "ir.png"):
            q = public / "test" / "b" / f.relative_to(frag)
            q.parent.mkdir(parents=True, exist_ok=True)
            os.link(f.resolve(), q)
    pd.DataFrame({"Id": ["b"], "Predicted": ["1 1 5 1"]}).to_csv(public / "sample_submission.csv", index=False)


PREPARES = {"vesuvius": _prepare_vesuvius, "gesture": _prepare_gesture, "jigsaw": _prepare_jigsaw, "hubmap": _prepare_hubmap, "whale": _prepare_whale, "billion": _prepare_billion,
            "jigsaw_ub": _prepare_jigsaw_ub, "freesound": _prepare_freesound}


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


def _hubmap_test_labels(pub: Path) -> None:
    """hubmap: MLE-bench copies each test slide's <id>.json (its glomerulus polygons: the answer) into public/test/;
    Kaggle's test folder had only the image and its anatomical-structure file. Removed, from MLE-bench's test slides
    too. test/sample_submission.csv (a copy prepare.py puts there) is rewritten as the merged sample."""
    import re

    for p in sorted((pub / "test").glob("*.json")):
        if re.fullmatch(r"[0-9a-f]+\.json", p.name):
            p.unlink()
            print(f"removed test label file {p.name}")
    q = pub / "test" / "sample_submission.csv"
    q.unlink(missing_ok=True)
    shutil.copyfile(pub / "sample_submission.csv", q)


def _gesture_drop_training3(pub: Path) -> None:
    (pub / "training3.tar.gz").unlink()


POSTS = {"gesture": _gesture_drop_training3, "siim": _strip_tfrecord_labels, "hubmap": _hubmap_test_labels,
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


def _make_herbarium(comp, out: Path) -> Path:
    """herbarium-2020: MLE-bench's per-category rule (1 image: train; <5: the last max(1, 20%) to test; else
    train_test_split(test_size=0.2, random_state=0)) applied to MLE-bench's train metadata, its validation images
    shuffled with random.Random(0) as it did. Its prepare.py numbers test images 0.. (images/<id // 1000>/<id>.jpg), so
    instead of running it again the validation images are numbered after MLE-bench's test images, in the same layout,
    and appended to test/metadata.json and sample_submission.csv."""
    import random

    from sklearn.model_selection import train_test_split

    pub0, nb = comp.public_dir, "nybg2020"
    with open(pub0 / nb / "train/metadata.json", encoding="latin-1") as f:
        meta = json.load(f)
    by_cat: dict = {}
    for a, im in zip(meta["annotations"], meta["images"]):
        assert a["image_id"] == im["id"]
        by_cat.setdefault(a["category_id"], []).append({"annotation": a, "image": im})
    keep, val = [], []
    for items in by_cat.values():
        n = len(items)
        if n == 1:
            tr, te = items, []
        elif n < 5:
            k = max(1, int(n * 0.2))
            tr, te = items[:-k], items[-k:]
        else:
            tr, te = train_test_split(items, test_size=0.2, random_state=0)
        keep += tr
        val += te
    random.Random(0).shuffle(val)
    with open(pub0 / nb / "test/metadata.json", encoding="latin-1") as f:
        test_meta = json.load(f)
    n_test = len(test_meta["images"])
    assert {int(im["id"]) for im in test_meta["images"]} == set(range(n_test))
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        pub, priv = Path(tmp) / "public", Path(tmp) / "private_val"
        (pub / nb / "train").mkdir(parents=True)
        priv.mkdir()
        _link_tree(pub0 / "description.md", pub / "description.md")
        _link_tree(pub0 / nb / "test/images", pub / nb / "test/images")  # MLE-bench's test images
        new_train = {**meta, "annotations": [x["annotation"] for x in keep], "images": [x["image"] for x in keep]}
        for im in new_train["images"]:
            q = pub / nb / "train" / im["file_name"]
            q.parent.mkdir(parents=True, exist_ok=True)
            os.link(pub0 / nb / "train" / im["file_name"], q)
        with open(pub / nb / "train/metadata.json", "w") as f:
            json.dump(new_train, f, indent=4, sort_keys=True)
        answers, sample = [], []
        for i, x in enumerate(val):
            vid = n_test + i
            im = {**x["image"], "id": str(vid), "file_name": f"images/{vid // 1000:03d}/{vid}.jpg"}
            q = pub / nb / "test" / im["file_name"]
            q.parent.mkdir(parents=True, exist_ok=True)
            os.link(pub0 / nb / "train" / x["image"]["file_name"], q)
            test_meta["images"].append(im)
            answers.append({"Id": str(vid), "Predicted": x["annotation"]["category_id"]})
        with open(pub / nb / "test/metadata.json", "w") as f:
            json.dump(test_meta, f, indent=4, sort_keys=True)
        val_ans = pd.DataFrame(answers)
        val_ans.to_csv(priv / "answers.csv", index=False)
        pd.concat([_read_table(pub0 / "sample_submission.csv"), val_ans.assign(Predicted="0")]).to_csv(
            pub / "sample_submission.csv", index=False)
        # leak checks: no validation image is a train' file or in train' metadata; every train' category kept
        val_files = {x["image"]["file_name"] for x in val}
        assert not val_files & {im["file_name"] for im in new_train["images"]}
        assert not any((pub / nb / "train" / f).exists() for f in val_files), "leak: validation image in train'"
        shutil.move(str(pub), out / "public")
        shutil.move(str(priv), out / "private_val")
    stats = {"competition": comp.id, "id_column": "Id", "validation_ids": len(val), "test_ids": n_test,
             "answers": "answers.csv", "seed": 0}
    (out / "valsplit.json").write_text(json.dumps(stats, indent=1) + "\n")
    (out / "DONE").touch()
    return out


CUSTOM = {"mlsp": _make_mlsp, "herbarium": _make_herbarium}


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
            if spec.json_id:
                hit |= _json_ids(p, spec.json_id) & _file_ids(val_ids, spec)
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
            if p.name in spec.keep_public:
                _link_tree(p, pub / p.name)
            elif p.name in spec.test_side:
                if p.name not in produced:
                    raise AssertionError(f"{cid}: prepare.py wrote no validation part of {p.name}")
                _merge(p, pub_v / p.name, pub / p.name, tmp, _file_ids(test_ids, spec), whole=p.name in spec.whole_test)
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


def require_pandas2() -> None:
    """MLE-bench's prepare.py scripts break under pandas 3 (`.unique()` returns an ArrowStringArray that
    train_test_split rejects): stop early instead of failing halfway through a competition."""
    if int(pd.__version__.split(".")[0]) >= 3:
        raise SystemExit(f"pandas {pd.__version__}: run this with the grader venv's python (pandas<3; "
                         "src/benchmarks/mlebench/setup_grader_venv.sh)")


def main() -> None:
    require_pandas2()
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
