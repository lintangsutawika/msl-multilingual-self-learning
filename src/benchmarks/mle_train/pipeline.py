#!/usr/bin/env python3
"""MLE training-data pipeline: sources -> MLE-bench-layout tasks -> public/private split -> Harbor tasks.

Run as a plain script with Python >= 3.11 (it does not import the parent package, so mlebench need not be
installed). Everything in one go, all outputs under one directory:

    python src/benchmarks/mle_train/pipeline.py all --workdir ~/mle-train            # needs a Kaggle API key
    python src/benchmarks/mle_train/pipeline.py all --workdir ~/mle-train --no-kaggle  # DSBench tasks only

    ~/mle-train/MLE-Dojo/   ~/mle-train/raw/   ~/mle-train/root/ (+ index.tsv)   ~/mle-train/harbor/   ~/mle-train/train.jsonl

Or step by step:

    P=src/benchmarks/mle_train/pipeline.py
    python $P mledojo-download --dojo-repo MLE-Dojo --data-dir DATA/mledojo [--max-gb 5] [--dsbench]
    python $P mledojo-rules    --dojo-repo MLE-Dojo --data-dir DATA/mledojo
    python $P mledojo-import   --dojo-repo MLE-Dojo --data-dir DATA/mledojo --root ROOT
    python $P sandmle-download --out DATA/sandmle --n-tasks 40 --max-gb 100
    python $P sandmle-build    --sample DATA/sandmle --root ROOT [--scale 35]
    python $P split            --root ROOT
    python $P harbor           --root ROOT --out TASKS [--source sandmle] [--limit 10]
    python $P export           --tasks-dir TASKS --out DATA/train.jsonl

See README.md for the format and the full walkthrough.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

if sys.version_info < (3, 11):
    sys.exit("pipeline.py needs Python >= 3.11")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harbor  # noqa: E402
import split  # noqa: E402
from sources import mledojo, sandmle  # noqa: E402


def _step(name: str) -> None:
    print(f"\n=== {name}", flush=True)


def run_all(a: argparse.Namespace) -> None:
    w = a.workdir.expanduser().resolve()
    raw, root, tasks = w / "raw/mledojo", w / "root", w / "harbor"
    root.mkdir(parents=True, exist_ok=True)
    _step("MLE-Dojo checkout")
    dojo = mledojo.clone(a.dojo_repo.expanduser().resolve() if a.dojo_repo else w / "MLE-Dojo")
    if not a.skip_download:
        _step("download (resumable; finished competitions are skipped)")
        mledojo.download(dojo, raw, a.max_gb, None, dsbench=not a.no_dsbench, kaggle=not a.no_kaggle)
    _step("import")
    mledojo.import_tasks(dojo, raw, root, None)
    _step("split + quality checks")
    split.run(root, None, a.workers, a.min_public_ids)
    _step("Harbor tasks")
    if tasks.exists():
        shutil.rmtree(tasks)  # rebuilt from scratch so tasks that stopped being usable disappear
    harbor.build(root, tasks, None, None, None, a.max_public_mb, a.agent_timeout, a.cpus, a.memory_mb)
    _step("export")
    harbor.export(tasks, w / "train.jsonl", None)
    links = mledojo.kaggle_rules_links(dojo, raw)
    if links:
        print(f"\n{len(links)} competitions were refused by Kaggle; accept their rules, then rerun this command:")
        print("\n".join(links))
    print(f"\nHARBOR_TASKS_DIR={tasks}\nPROMPT_DATA={w / 'train.jsonl'}\nindex (why each task is in or out): "
          f"{root / 'index.tsv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))

    p = sub.add_parser("all", help="clone MLE-Dojo, download, import, split, build Harbor tasks, export")
    p.add_argument("--workdir", type=Path, required=True, help="everything goes here")
    p.add_argument("--dojo-repo", type=Path, help="existing MLE-Dojo checkout (default: clone into WORKDIR)")
    p.add_argument("--no-kaggle", action="store_true", help="skip Kaggle competitions; build DSBench tasks only")
    p.add_argument("--no-dsbench", action="store_true", help="skip DSBench tasks")
    p.add_argument("--skip-download", action="store_true", help="reuse WORKDIR/raw as is")
    p.add_argument("--max-gb", type=float, default=5.0, help="skip Kaggle competitions whose zip is larger")
    p.add_argument("--workers", type=int, default=cpus)
    p.add_argument("--min-public-ids", type=int, default=100)
    p.add_argument("--max-public-mb", type=float, default=2000.0)
    p.add_argument("--agent-timeout", type=int, default=7200)
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--memory-mb", type=int, default=16384)

    p = sub.add_parser("mledojo-download", help="download + prepare MLE-Dojo competitions (Kaggle) and DSBench")
    p.add_argument("--dojo-repo", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--max-gb", type=float, default=5.0, help="skip competitions whose zip is larger")
    p.add_argument("--competitions", nargs="*", help="only these (ignores --max-gb)")
    p.add_argument("--dsbench", action="store_true", help="also prepare MLE-Dojo's DSBench tasks")
    p.add_argument("--no-kaggle", action="store_true", help="skip the Kaggle competitions")

    p = sub.add_parser("mledojo-rules", help="print Kaggle rules links for competitions that failed to download")
    p.add_argument("--dojo-repo", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)

    p = sub.add_parser("mledojo-import", help="MLE-Dojo prepared data -> ROOT/<task>/ (MLE-bench layout)")
    p.add_argument("--dojo-repo", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--tasks", nargs="*")

    p = sub.add_parser("sandmle-download", help="stream the first N SandMLE tasks from the Drive archive")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-tasks", type=int, default=40)
    p.add_argument("--max-gb", type=float, default=100.0)

    p = sub.add_parser("sandmle-build", help="regenerate SandMLE tasks at a larger size -> ROOT/<task>/")
    p.add_argument("--sample", type=Path, required=True, help="dir with <task>/prepared/public/ from sandmle-download")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--scale", type=int, default=35, help="multiply the generator's 147/137/127 sample count")
    p.add_argument("--max-sample-mb", type=float, default=25.0, help="skip tasks larger than this as shipped")
    p.add_argument("--timeout", type=int, default=3600, help="seconds per generator run")
    p.add_argument("--workers", type=int, default=cpus)
    p.add_argument("--tasks", nargs="*")

    p = sub.add_parser("split", help="public/private split + grading sanity check -> config.yaml, ROOT/index.tsv")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--tasks", nargs="*")
    p.add_argument("--workers", type=int, default=cpus)
    p.add_argument("--min-public-ids", type=int, default=100)

    p = sub.add_parser("harbor", help="usable tasks -> Harbor task dirs")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tasks", nargs="*")
    p.add_argument("--source", help="only this source (mledojo-kaggle, mledojo-dsbench, sandmle)")
    p.add_argument("--limit", type=int, help="at most N tasks (smallest public data first)")
    p.add_argument("--max-public-mb", type=float, default=2000.0)
    p.add_argument("--agent-timeout", type=int, default=7200)
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--memory-mb", type=int, default=16384)

    p = sub.add_parser("export", help="Harbor tasks -> trainer prompt jsonl, metric_directions.json, dispatch + per-source task lists")
    p.add_argument("--tasks-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--agent-name", help="optional metadata.agent_name for the rollout function")

    a = ap.parse_args()
    if a.cmd == "all":
        run_all(a)
    elif a.cmd == "mledojo-download":
        mledojo.download(a.dojo_repo.resolve(), a.data_dir.resolve(), a.max_gb, a.competitions, a.dsbench,
                         kaggle=not a.no_kaggle)
    elif a.cmd == "mledojo-rules":
        print("\n".join(mledojo.kaggle_rules_links(a.dojo_repo, a.data_dir)))
    elif a.cmd == "mledojo-import":
        a.root.mkdir(parents=True, exist_ok=True)
        mledojo.import_tasks(a.dojo_repo.resolve(), a.data_dir.resolve(), a.root.resolve(), a.tasks)
    elif a.cmd == "sandmle-download":
        sandmle.download(a.out, a.n_tasks, a.max_gb)
    elif a.cmd == "sandmle-build":
        sandmle.build(a.sample.resolve(), a.root.resolve(), a.scale, a.max_sample_mb, a.timeout, a.workers, a.tasks)
    elif a.cmd == "split":
        split.run(a.root.resolve(), a.tasks, a.workers, a.min_public_ids)
    elif a.cmd == "harbor":
        harbor.build(a.root.resolve(), a.out.resolve(), a.tasks, a.source, a.limit, a.max_public_mb,
                     a.agent_timeout, a.cpus, a.memory_mb)
    elif a.cmd == "export":
        harbor.export(a.tasks_dir.resolve(), a.out.resolve(), a.agent_name)


if __name__ == "__main__":
    main()
