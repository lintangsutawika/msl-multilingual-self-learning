"""Prepare LeetCode Harbor tasks + rollout.jsonl for Miles GRPO training.

Modeled on mho's DAPO-Math prepare (tasks/dapo_math_17k/main.py) but sources the
flat per-(problem, language) rows from the hosted ``neulab/leetcode`` train/test
split (via the msl dataset loader) and renders Harbor task dirs through the msl
adapter (adapter.generate_all), which uses the per-language templates (+ optional
prebuilt sifs via --images).

Consumed by scripts/optimize/train.sh:
  - builds Harbor task dirs (one per (problem, language)) under --output-dir
    (the MATH_TRAIN_DIR used as HARBOR_TASKS_DIR by the mile connector), and
  - emits a ``rollout.jsonl`` of {prompt, metadata, ...} lines the Miles trainer
    reads with ``--prompt-data`` / ``--input-key prompt`` / ``--metadata-key metadata``.

Usage:
    uv run -m src.benchmarks.leetcode.hf_dataset.tasks.leetcode \
        --split train --max-tasks 500 --output-dir /path/to/train \
        --emit-rollout-only [--skip-unsupported] [--images /dir/<lang>.sif ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running as a module or script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.benchmarks.leetcode.dataset import load_problems_hf  # noqa: E402
from src.benchmarks.leetcode.adapter import generate_all as adapter_generate_all  # noqa: E402

SPLIT = "train"


def _prompt_for(row: dict[str, Any]) -> str:
    """Build the model prompt for a (problem, language) row: the LeetCode problem
    statement plus a directive to write a complete <<language>> source file."""
    lang = row["language"]
    return (
        f"{row['problem_description'].strip()}\n\n"
        f"Solve this LeetCode problem in {lang}. Write a complete source file, "
        f"including any required imports, includes, or package declarations. Do "
        f"not define a program entrypoint. Your solution must compile or run as "
        f"submitted.\n\n"
        f"Save your final solution to the source file in the task working "
        f"directory for {lang}."
    )


def _rollout_line(row: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """One rollout.jsonl record: the prompt + metadata the Miles trainer consumes."""
    return {
        "prompt": _prompt_for(row),
        "metadata": {
            **(metadata or {}),
            "question_id": row["question_id"],
            "task_id": row["task_id"],
            "language": row["language"],
        },
    }


def emit_rollout(rows: list[dict[str, Any]], out: Path) -> None:
    """Write {prompt, metadata} lines the Miles trainer reads as --prompt-data."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_rollout_line(row)) + "\n")
    print(f"Wrote {len(rows)} rollout records to {out}")


def build_tasks(
    rows: list[dict[str, Any]],
    output: Path,
    images: dict[str, str] | None = None,
    skip_unsupported: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Render Harbor task dirs via the msl adapter. Returns (exclusions, count)."""
    return adapter_generate_all(rows, output=output, images=images,
                                skip_unsupported=skip_unsupported)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare LeetCode Harbor tasks + rollout.jsonl.")
    ap.add_argument("--split", default=SPLIT, help="neulab/leetcode split (train|test).")
    ap.add_argument("--max-tasks", type=int, default=0,
                    help="Cap the number of (problem, language) rows (0 = all).")
    ap.add_argument("--start", type=int, default=0,
                    help="Row offset to start from (for contiguous sharding).")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Where task dirs are written (default: tasks/leetcode-<split>).")
    ap.add_argument("--emit-rollout-only", action="store_true",
                    help="Only (re)write rollout.jsonl from existing task iterating rows; no task gen.")
    ap.add_argument("--skip-unsupported", action="store_true",
                    help="Record unsupported (object-transport) rows in exclusions.json.")
    ap.add_argument("--images", default="", help="Comma-separated lang=sif pairs (e.g. python=/p.sif,cpp=/c.sif).")
    args = ap.parse_args()

    # Load the flat per-(problem, language) rows for the split.
    rows = load_problems_hf(split=args.split)
    if args.max_tasks and args.max_tasks > 0:
        rows = rows[args.start : args.start + args.max_tasks]

    output_dir = args.output_dir or Path(f"tasks/leetcode-{args.split}")

    images = {}
    for tok in filter(None, (t.strip() for t in args.images.split(","))):
        lang, _, sif = tok.partition("=")
        if lang and sif:
            images[lang] = sif

    if args.emit_rollout_only:
        emit_rollout(rows, output_dir / "rollout.jsonl")
        return

    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}. Use a fresh --output-dir.")
    exclusions, count = build_tasks(rows, output_dir, images=images or None,
                                    skip_unsupported=args.skip_unsupported)
    emit_rollout(rows, output_dir / "rollout.jsonl")
    print(f"Generated {count} tasks in {output_dir}; excluded {len(exclusions)}")


if __name__ == "__main__":
    main()