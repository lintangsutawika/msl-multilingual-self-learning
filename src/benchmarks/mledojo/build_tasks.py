"""Build the MLE-Dojo split tasks (Harbor, SAIL: data baked into the image) from the validation splits.

The same layout as the MLE-bench split tasks (src/benchmarks/mlebench/make_split_tasks.py, baked mode):

    <out>/<task>/
        instruction.md, task.toml
        environment/public/        valsplit/public: train' + the test inputs with the validation inputs shuffled in
        environment/private_val/   the validation answers: /workspace/private_val in the box, root-only
        environment/grader/        the task's MLE-Dojo grader + grade_cli.py (/opt/grader, root-only)
        environment/leaderboard.csv  Kaggle's private leaderboard, for the trainer (not copied into the image)
        environment/score_key      per-task HMAC key for validate's signed scores (SCORE_KEYS_DIR = <out>)
        tests/private/answers.csv  the TEST answers (MLE-Dojo's test set, untouched): Harbor uploads tests/ only
                                   after the agent finishes
        tests/grader/, tests/leaderboard.csv, tests/test.sh   the final grade -> verifier/grade.json
        solution/solve.sh          oracle: submits the sample submission

    python -m src.benchmarks.mledojo.build_tasks --data DIR --out OUT [--task a,b] [--keys-from OLD_OUT]

DIR: one folder per task with config.yaml, grader/, leaderboard.csv, prepared/ and valsplit/ (valsplit.py). Data is
hardlinked from DIR (OUT must be on the same filesystem). Files above 4.9 GiB (SAIL's limit) are shipped gzipped.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import secrets
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "task-template"
MAX_FILE_BYTES = int(4.9 * 2**30)


def _fill(path: Path, kw: dict) -> None:
    """Replace {key} for known keys in one pass (unknown braces, e.g. in f-strings, are left alone)."""
    path.write_text(re.sub(r"\{(\w+)\}", lambda m: str(kw[m.group(1)]) if m.group(1) in kw else m.group(0),
                           path.read_text()))


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


def _gzip_big_files(task: Path, public: Path) -> None:
    """Replace each public file above MAX_FILE_BYTES with a gzipped copy, cached in valsplit/_gz/ (rebuilds reuse it).
    Kaggle shipped such files gzipped too (acquire-valued-shoppers' transactions, avazu's train); pandas reads them."""
    for f in sorted(p for p in public.rglob("*") if p.is_file() and p.stat().st_size > MAX_FILE_BYTES):
        rel = f.relative_to(public)
        src = task / "valsplit/public" / rel
        cache = task / "valsplit/_gz" / (str(rel) + ".gz")
        if not cache.exists() or cache.stat().st_mtime < src.stat().st_mtime:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp")
            with open(src, "rb") as i, gzip.open(tmp, "wb", compresslevel=6) as o:
                shutil.copyfileobj(i, o, 64 << 20)
            os.replace(tmp, cache)
        if cache.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"{rel} is still {cache.stat().st_size / 2**30:.1f} GiB gzipped")
        f.unlink()
        os.link(cache, f.with_name(f.name + ".gz"))


def build_one(task: Path, out: Path, keys_from: Path | None, agent_timeout: int, cpus: int, memory_mb: int) -> Path:
    import yaml

    if not (task / "valsplit/DONE").exists():
        raise ValueError(f"{task.name}: no validation split (run valsplit.py first)")
    cfg = yaml.safe_load((task / "config.yaml").read_text())
    vs = json.loads((task / "valsplit/valsplit.json").read_text())
    if not (task / "leaderboard.csv").exists():
        raise ValueError(f"{task.name}: no leaderboard.csv (every training task's final grade is medal-based)")
    t = out / task.name
    key = t / "environment/score_key"
    old = keys_from / task.name / "environment/score_key" if keys_from else None
    old_key = key.read_text() if key.exists() else old.read_text() if old and old.exists() else None
    if t.exists():
        shutil.rmtree(t)
    shutil.copytree(TEMPLATE, t, ignore=shutil.ignore_patterns("__pycache__"))
    key.write_text(old_key or secrets.token_hex(32) + "\n")
    key.chmod(0o600)
    env, tests = t / "environment", t / "tests"
    _link_tree(task / "valsplit/public", env / "public")
    _gzip_big_files(task, env / "public")
    _link_tree(task / "valsplit/private_val/answers.csv", env / "private_val/answers.csv")
    _link_tree(task / "prepared/private/answers.csv", tests / "private/answers.csv")
    _link_tree(task / "leaderboard.csv", env / "leaderboard.csv")
    _link_tree(task / "leaderboard.csv", tests / "leaderboard.csv")
    for g in (env / "grader", tests / "grader"):
        shutil.copytree(task / "grader", g, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(HERE / "grade_cli.py", g / "grade_cli.py")
    desc = task / "prepared/public/description.md"
    hib = bool(cfg["higher_is_better"])
    kw = {
        "task_id": task.name,
        "source": cfg["source"],
        "higher_is_better": str(hib).lower(),
        "id_column": vs["id_column"],
        "validation_ids": vs["validation_ids"],
        "test_ids": vs["test_ids"],
        "valsplit_rule": vs["rule"],
        "reward": cfg.get("reward", "leaderboard_position"),
        "agent_timeout": float(agent_timeout),
        "cpus": cpus,
        "memory_mb": memory_mb,
        "memory_gb": memory_mb // 1024,
        "hours": f"{agent_timeout / 3600:g}",
        "sample_submission": cfg["sample_submission"],
        "direction": "higher is better" if hib else "lower is better",
        "description": desc.read_text(errors="replace").strip() if desc.exists() else f"Task {task.name}.",
    }
    for rel in ("task.toml", "instruction.md", "environment/validate.py", "tests/test.sh", "solution/solve.sh"):
        _fill(t / rel, kw)
    for rel in ("environment/bin/validate", "environment/bin/validate-root", "tests/test.sh", "solution/solve.sh"):
        (t / rel).chmod(0o755)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", help="comma-separated (default: every task with a finished split)")
    ap.add_argument("--keys-from", help="reuse each task's score_key from this earlier build (e.g. the old harbor/)")
    ap.add_argument("--agent-timeout", type=int, default=7200)
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--memory-mb", type=int, default=16384)
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = a.task.split(",") if a.task else sorted(p.parent.parent.name for p in data.glob("*/valsplit/DONE"))
    bad = 0
    for n in names:
        try:
            print("built", build_one(data / n, out, Path(a.keys_from) if a.keys_from else None, a.agent_timeout,
                                     a.cpus, a.memory_mb), flush=True)
        except Exception as e:
            bad += 1
            print(f"FAILED {n}: {e}", flush=True)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
