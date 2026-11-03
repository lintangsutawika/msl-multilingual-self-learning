"""MLE-Dojo source: real Kaggle competitions (and DSBench's Kaggle-derived tasks) prepared by MLE-Dojo's own code.

clone:    MLE-Dojo at the commit this pipeline was validated with (MLEDOJO_COMMIT).
download: MLE-Dojo's prepare/mle.py:prepare_competition per competition (needs Kaggle credentials and the
          competition's rules accepted on kaggle.com), smallest zip first, MLE-bench competitions skipped,
          raw/ deleted after preparing; then optionally prepare/dsbench.py (no Kaggle needed).
          Writes <data_dir>/kaggle/_status.tsv.
import:   <data_dir>/{kaggle,dsbench/prepared}/<id>/data/{public,private} -> <root>/<id>/ (MLE-bench layout).
          The grader is the competition's MLE-Dojo metric class; the leaderboard is Kaggle's private leaderboard.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from common import GRADER_SRC, find_sample, is_mlebench, link_file, link_tree, write_config

MLEDOJO_URL = "https://github.com/MLE-Dojo/MLE-Dojo"
MLEDOJO_COMMIT = "667a8fd0e959eb47f140ce7e621f831fdb55e5dd"  # validated 2026-11-02

UNITS = {"B": 1, "kB": 1e3, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}


def _zip_bytes(info: dict) -> float:
    m = re.match(r"([\d.,]+)\s*(\w+)", info.get("data_info", {}).get("size", ""))
    return float(m.group(1).replace(",", "")) * UNITS.get(m.group(2), 0) if m else float("inf")


def clone(dest: Path) -> Path:
    """Clone MLE-Dojo at the validated commit (no-op if dest already exists)."""
    if not (dest / "prepare/mle.py").exists():
        subprocess.run(["git", "clone", "--quiet", MLEDOJO_URL, str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", MLEDOJO_COMMIT], check=True)
    head = _git_commit(dest)
    if head and head != MLEDOJO_COMMIT:
        print(f"warning: {dest} is at {head[:8]}, the pipeline was validated with {MLEDOJO_COMMIT[:8]}")
    return dest


def kaggle_credentials_ok() -> bool:
    cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    return cfg.exists() or bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))


def kaggle_rules_links(dojo_repo: Path, data_dir: Path) -> list[str]:
    """Rules pages to accept for competitions whose download was refused (status rules_or_download)."""
    st = data_dir / "kaggle/_status.tsv"
    rows = [l.split("\t") for l in st.read_text().splitlines()[1:]] if st.exists() else []
    return [f"https://www.kaggle.com/competitions/{r[0]}/rules" for r in rows if r[1] == "rules_or_download"]


def download(dojo_repo: Path, data_dir: Path, max_gb: float, competitions: list[str] | None, dsbench: bool,
             kaggle: bool = True) -> None:
    # MLE-Dojo shells out to the `kaggle` CLI: make the one installed next to this Python findable without
    # requiring the venv to be activated.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    if kaggle:
        if not kaggle_credentials_ok():
            raise SystemExit("no Kaggle credentials: put kaggle.json in ~/.kaggle (or set KAGGLE_USERNAME and "
                             "KAGGLE_KEY), or pass --no-kaggle to build only the DSBench tasks")
        if not shutil.which("kaggle"):
            raise SystemExit("the `kaggle` CLI is not installed: pip install -r requirements.txt")
        _download_kaggle(dojo_repo, data_dir, max_gb, competitions)
    if dsbench:
        _download_dsbench(dojo_repo, data_dir)


def _download_kaggle(dojo_repo: Path, data_dir: Path, max_gb: float, competitions: list[str] | None) -> None:
    sys.path[:0] = [str(dojo_repo), str(dojo_repo / "prepare")]
    from mle import prepare_competition  # MLE-Dojo's own downloader + per-competition prepare.py

    kdir = data_dir / "kaggle"
    logs = kdir / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(dojo_repo / "prepare/mle.json"))
    comps = sorted((_zip_bytes(v), k) for k, v in meta.items() if not is_mlebench(k) and (
        (competitions and k in competitions) or (not competitions and _zip_bytes(v) <= max_gb * 1e9)))
    status = kdir / "_status.tsv"
    rows = {l.split("\t")[0]: l for l in status.read_text().splitlines()[1:]} if status.exists() else {}
    for sz, cid in comps:
        pub, priv = kdir / cid / "data/public", kdir / cid / "data/private"
        if rows.get(cid, "").split("\t")[1:2] == ["ok"] and pub.exists() and any(priv.glob("*")):
            continue  # done in an earlier run (the job is resumable)
        t0 = time.time()
        print(f"=== {time.strftime('%H:%M:%S')} {cid} ({sz / 1e9:.2f} GB zip)", flush=True)
        # A few MLE-Dojo prepare scripts use unseeded randomness (e.g. the-nature-conservancy-fisheries-monitoring
        # shuffles its test split, nyc-taxi-trip-duration draws a random sample submission). Seed per competition
        # so every machine builds the same data.
        random.seed(zlib.crc32(cid.encode()))
        np.random.seed(zlib.crc32(cid.encode()))
        ok, err = prepare_competition(cid, kdir, logs)
        log = (logs / f"{cid}.log").read_text(errors="replace") if (logs / f"{cid}.log").exists() else ""
        if ok:
            st = "ok"
        elif re.search(r"403|Forbidden|must accept|rules", log, re.I) or ".zip" in str(err):
            st = "rules_or_download"
        else:
            st = "prepare_error"
        shutil.rmtree(kdir / cid / "raw", ignore_errors=True)
        rows[cid] = f"{cid}\t{st}\t{sz / 1e9:.3f}\t{time.time() - t0:.0f}\t{'' if ok else (err or '')[:200]}".replace("\n", " ")
        print(rows[cid], flush=True)
        status.write_text("competition\tstatus\tzip_gb\tseconds\tnote\n"
                          + "\n".join(rows[c] for _, c in comps if c in rows) + "\n")


def _download_dsbench(dojo_repo: Path, data_dir: Path) -> None:
    """MLE-Dojo's DSBench preparation (data from Hugging Face, no Kaggle) with a robust extraction step."""
    if (data_dir / "dsbench/.done").exists():
        return
    raw = data_dir / "dsbench/raw"
    if (raw / "data").exists():
        shutil.rmtree(raw / "data")  # half-extracted leftovers from a failed attempt
    r = subprocess.run([sys.executable, str(Path(__file__).with_name("dsbench_prepare.py")), str(dojo_repo),
                        str(raw), str(data_dir / "dsbench/prepared")], stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError(f"DSBench preparation failed (rc={r.returncode}); see the output above")
    (data_dir / "dsbench/.done").touch()
    shutil.rmtree(raw, ignore_errors=True)


def _sources(data_dir: Path):
    st = data_dir / "kaggle/_status.tsv"
    ok = [l.split("\t")[0] for l in st.read_text().splitlines()[1:] if l.split("\t")[1] == "ok"] if st.exists() else []
    for cid in ok:
        yield "mledojo-kaggle", cid, data_dir / "kaggle" / cid / "data"
    ds = data_dir / "dsbench/prepared"
    for cid in sorted(os.listdir(ds)) if ds.exists() else []:
        yield "mledojo-dsbench", cid, ds / cid / "data"


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def import_tasks(dojo_repo: Path, data_dir: Path, root: Path, only: list[str] | None) -> None:
    commit = _git_commit(dojo_repo)
    seen = set()
    for source, cid, data in _sources(data_dir):
        if (only and cid not in only) or cid in seen:
            continue  # a Kaggle copy wins over a DSBench copy of the same competition
        seen.add(cid)
        if is_mlebench(cid):
            shutil.rmtree(root / cid, ignore_errors=True)
            print(f"skip {cid}: MLE-bench competition (eval set)")
            continue
        metric = dojo_repo / "mledojo/competitions" / cid / "utils/metric.py"
        answers = data / "private/test_answer.csv"
        sample = find_sample(data / "public") if (data / "public").exists() else None
        if not (metric.exists() and answers.exists() and sample):
            print(f"skip {cid}: metric={metric.exists()} answers={answers.exists()} sample={bool(sample)}")
            continue
        t = root / cid
        if t.exists():
            shutil.rmtree(t)
        link_tree(data / "public", t / "prepared/public")
        desc = data / "public/description.txt"
        if desc.exists() and not (t / "prepared/public/description.md").exists():
            shutil.copy(desc, t / "prepared/public/description.md")
        link_file(answers, t / "prepared/private/answers.csv")
        lb = data / "private/private_leaderboard.csv"
        if lb.exists():
            link_file(lb, t / "leaderboard.csv")
        g = t / "grader"
        (g / "mledojo/metrics").mkdir(parents=True)
        shutil.copy(GRADER_SRC / "mledojo_grade.py", g / "grade.py")
        shutil.copy(metric, g / "metric.py")
        shutil.copy(dojo_repo / "mledojo/metrics/base.py", g / "mledojo/metrics/base.py")
        for f in ("mledojo/__init__.py", "mledojo/metrics/__init__.py"):
            (g / f).touch()
        write_config(t, {
            "id": cid,
            "name": cid,
            "source": source,
            "origin": f"https://www.kaggle.com/competitions/{cid}",
            "provenance": {"mledojo_commit": commit, "data": str(data), "imported": time.strftime("%Y-%m-%d")},
            "sample_submission": sample.name,
            "id_column": None,  # filled in by `split` (first column of the sample submission)
            "leaderboard": "leaderboard.csv" if lb.exists() else None,
            "reward": "leaderboard_position" if lb.exists() else "normalized",
        })
        print(f"imported {source:16s} {cid}")
