"""Prepare as many MLE-bench competitions as the disk allows: by split (low, medium, high), smallest download first.

    $GRADER_PY -m src.benchmarks.mlebench.prepare_queue [--splits low,medium,high | --competition a,b]
        [--reserve-gb 200] [--delete-zips]

Run it with the grader venv (setup_grader_venv.sh): MLE-bench's prepare.py scripts break under pandas 3. Needs Kaggle
credentials (KAGGLE_CONFIG_DIR) and each competition's rules accepted on kaggle.com.

For each competition that isn't prepared yet: look up its Kaggle download size, skip it if preparing it could leave
less than --reserve-gb free (estimate: 3x the download, for the zip, the extraction and the prepared copy), else
download + prepare it with MLE-bench's own scripts (adapter.prepare). With --delete-zips, Kaggle's zip and its extraction (raw/) are
removed once the competition is prepared (MLE-bench keeps it; it is only needed to prepare again). A competition whose
Kaggle rules are not accepted is recorded and skipped; rerun after accepting them (finished ones are skipped).
Status: <mlebench data dir>/_prepare_queue.tsv.
"""
from __future__ import annotations

import argparse
import shutil
import time
import traceback
from pathlib import Path

from mlebench.data import is_dataset_prepared
from mlebench.registry import registry

from .adapter import prepare
from .valsplit import require_pandas2


def _kaggle_bytes(api, cid: str) -> int | None:
    """Size of the competition's download zip, from the download's Content-Length (nothing is downloaded). The file
    listing (competition_list_files) returns only its first page, so its sizes are far too small for big datasets."""
    try:
        resp = api.competitions_data_download_files_with_http_info(id=cid, _preload_content=False)[0]
        n = resp.headers.get("Content-Length")
        resp.close()
        return int(n) if n else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--splits", default="low,medium,high")
    ap.add_argument("--competition", help="comma-separated ids: prepare just these (instead of --splits)")
    ap.add_argument("--reserve-gb", type=float, default=200.0)
    ap.add_argument("--delete-zips", action="store_true", help="remove Kaggle's zip and raw/ after preparing")
    a = ap.parse_args()
    require_pandas2()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    data = registry.get_data_dir()
    status = data / "_prepare_queue.tsv"
    rows = {}
    if status.exists():
        for line in status.read_text().splitlines()[1:]:
            rows[line.split("\t")[0]] = line
    splits_dir = registry.get_splits_dir()
    groups = {s: (splits_dir / f"{s}.txt").read_text().split() for s in a.splits.split(",")}
    if a.competition:
        level = {c: s for s in ("low", "medium", "high") for c in (splits_dir / f"{s}.txt").read_text().split()}
        groups = {}
        for c in a.competition.split(","):
            groups.setdefault(level.get(c.strip(), "other"), []).append(c.strip())
    order = []
    for s, group in groups.items():
        ids = [c for c in group if c not in {c2 for _, c2, _ in order}]
        todo = [c for c in ids if not is_dataset_prepared(registry.get_competition(c))]
        sized = sorted((_kaggle_bytes(api, c) or float("inf"), c) for c in todo)
        order += [(s, c, b) for b, c in sized]
    print(f"{len(order)} competitions to prepare", flush=True)

    def save():
        status.write_text("competition\tsplit\tstatus\tkaggle_gb\tseconds\tnote\n" + "\n".join(rows.values()) + "\n")

    for split, cid, size in order:
        free = shutil.disk_usage(data).free
        gb = size / 1e9 if size != float("inf") else float("nan")
        if size == float("inf"):
            st, note, t = "rules_or_unknown_size", "Kaggle file list unavailable (rules not accepted?)", 0
        elif free - 3 * size < a.reserve_gb * 1e9:
            st, note, t = "skipped_no_space", f"{free / 1e9:.0f} GB free, needs ~{3 * gb:.0f} + {a.reserve_gb:.0f} reserve", 0
        else:
            t0 = time.time()
            print(f"=== {time.strftime('%H:%M:%S')} {split} {cid} ({gb:.1f} GB)", flush=True)
            try:
                prepare(cid)
                st, note = "ok", ""
                if a.delete_zips:  # prepared = public/ + private/; the zip and its extraction are only for re-preparing
                    for z in (data / cid).glob("*.zip"):
                        z.unlink()
                    shutil.rmtree(data / cid / "raw", ignore_errors=True)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                st = "rules" if ("403" in msg or "rules" in msg.lower()) else "error"
                note = msg[:200].replace("\t", " ").replace("\n", " ")
                traceback.print_exc()
            t = time.time() - t0
        rows[cid] = f"{cid}\t{split}\t{st}\t{gb:.2f}\t{t:.0f}\t{note}"
        print(rows[cid], flush=True)
        save()
    print("QUEUE-DONE", flush=True)


if __name__ == "__main__":
    main()
