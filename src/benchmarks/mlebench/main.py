"""CLI for generating MLE-Bench Harbor tasks from the shared template (task-template/).

The competition's data must be prepared first (prepare_queue.py, under the grader venv) and, for --val-split, split
(valsplit.py). See README.md in this directory for the whole pipeline. Two kinds of task:

    # Babel / Singularity: data mounted read-only at run time, answers kept out of the image (run with run_babel.sh)
    uv run python -m src.benchmarks.mlebench --competition spooky-author-identification --val-split --mount-data \
        --output /path/to/tasks

    # SAIL / Docker: prepared data baked into each task's image
    uv run python -m src.benchmarks.mlebench --splits low --val-split --output /path/to/tasks

    # Selecting competitions: --competition a,b | --splits low,medium,high | --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mlebench.registry import registry

from .adapter import generate_all

SPLIT_DIFFICULTY = {"low": "low", "medium": "medium", "high": "high"}
DEFAULT_OUTPUT = Path("harbor_tasks")


def _load_splits() -> dict[str, list[str]]:
    """Load competition ids per split from the mlebench experiments/splits dir."""
    splits_dir = registry.get_splits_dir()
    out: dict[str, list[str]] = {}
    for name in SPLIT_DIFFICULTY:
        p = splits_dir / f"{name}.txt"
        if p.exists():
            out[name] = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    return out


def _resolve_targets(args) -> dict[str, str]:
    """Return {competition_id: difficulty} to generate."""
    splits = _load_splits()
    difficulty: dict[str, str] = {}
    for sp, ids in splits.items():
        for i in ids:
            difficulty.setdefault(i, SPLIT_DIFFICULTY[sp])

    if args.competition:
        return {c.strip(): difficulty.get(c.strip(), "low") for c in args.competition.split(",") if c.strip()}
    if args.all:
        return {i: difficulty.get(i, "low") for i in registry.list_competition_ids()}
    sel = [s.strip() for s in args.splits.split(",") if s.strip()]
    targets: dict[str, str] = {}
    for s in sel:
        for i in splits.get(s, []):
            targets.setdefault(i, SPLIT_DIFFICULTY[s])
    return targets


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate MLE-Bench Harbor task dirs.")
    ap.add_argument("--splits", default="low", help="comma list: low,medium,high (default low)")
    ap.add_argument("--all", action="store_true", help="generate all registered competitions")
    ap.add_argument("--competition", help="competition id(s), comma-separated")
    ap.add_argument("--prepare", action="store_true",
                    help="download + prepare missing competition data first (needs Kaggle credentials)")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT),
                    help="output dir (default harbor_tasks); written under mlebench-<split>/ subdirs")
    ap.add_argument("--val-split", action="store_true",
                    help="carve a validation set out of train with MLE-bench's own split (valsplit.py): validate "
                         "scores it, MLE-bench's test stays the final grade")
    ap.add_argument("--mount-data", action="store_true",
                    help="don't copy the data into the task: the public data is bind-mounted read-only at run time "
                         "(task.toml metadata.public_data) and the test answers go to tests/ (Babel / Singularity)")
    args = ap.parse_args()

    output = Path(args.output)
    targets = _resolve_targets(args)
    made, failed = generate_all(targets, output, prepare_data=args.prepare, val_split=args.val_split, mount_data=args.mount_data)

    for cid, diff, d in made:
        print(f"generated {diff:6s} {cid} -> {d}")
    for cid, err in failed:
        print(f"FAILED {cid}: {err}")
    print(f"done: {len(made)} tasks under {output} (subdirs mlebench-low/medium/high), {len(failed)} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()