"""CLI for generating MLE-Bench Harbor tasks from the shared template.

Each task gets its competition's prepared data baked in (environment/{public,private,
leaderboard.csv}). Data comes from the mlebench cache (~/.cache/mle-bench/data);
--prepare downloads and prepares missing competitions first (needs Kaggle credentials,
e.g. KAGGLE_CONFIG_DIR=~/.kaggle). Run on the login node; it is CPU/network only.

Examples:
    # The 8 low-complexity comps the loop uses (prepare + bake)
    C=tabular-playground-series-may-2022,tabular-playground-series-dec-2021,spooky-author-identification
    C=$C,random-acts-of-pizza,new-york-city-taxi-fare-prediction,leaf-classification
    C=$C,denoising-dirty-documents,plant-pathology-2020-fgvc7
    uv run python -m src.benchmarks.mlebench --prepare --competition $C

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
    args = ap.parse_args()

    output = Path(args.output)
    targets = _resolve_targets(args)
    made, failed = generate_all(targets, output, prepare_data=args.prepare)

    for cid, diff, d in made:
        print(f"generated {diff:6s} {cid} -> {d}")
    for cid, err in failed:
        print(f"FAILED {cid}: {err}")
    print(f"done: {len(made)} tasks under {output} (subdirs mlebench-low/medium/high), {len(failed)} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()