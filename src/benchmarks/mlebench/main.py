"""CLI for generating MLE-Bench Harbor tasks from the shared template.

Examples:
    # Prepared benchmark sets
    uv run python -m src.benchmarks.mlebench --splits low
    uv run python -m src.benchmarks.mlebench --splits low,medium,high
    uv run python -m src.benchmarks.mlebench --all

    # Single competition
    uv run python -m src.benchmarks.mlebench --competition nomad2018-predict-transparent-conductors

    # Custom output dir
    uv run python -m src.benchmarks.mlebench --splits low --output /tmp/mlebench-tasks
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mlebench.registry import registry

from .adapter import generate_all

SPLIT_DIFFICULTY = {"low": "low", "medium": "medium", "high": "high"}
DEFAULT_OUTPUT = Path("benchmarks/mlebench/tasks")


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
        return {args.competition: SPLIT_DIFFICULTY.get(args.competition, "low")}
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
    ap.add_argument("--competition", help="single competition id")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT),
                    help="output dir (default benchmarks/mlebench/tasks); written under mlebench-<split>/ subdirs")
    args = ap.parse_args()

    output = Path(args.output)
    targets = _resolve_targets(args)
    made = generate_all(targets, output)

    for cid, diff, d in made:
        print(f"generated {diff:6s} {cid} -> {d}")
    print(f"done: {len(made)} tasks under {output} (subdirs mlebench-low/medium/high)")


if __name__ == "__main__":
    main()