"""CLI: generate Harbor leetcode tasks from the dataset + per-language templates.

Usage:
    uv run python -m src.benchmarks.leetcode --limit 10 --lang all
    uv run python -m src.benchmarks.leetcode --lang python --question-id 3243
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .adapter import LANGUAGES, generate_all

def load_problems(dataset: Path) -> list[dict]:
    return [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]

DEFAULT_DATASET = Path("benchmarks/leetcode/data/leetcode_multilingual.jsonl")
DEFAULT_OUTPUT = Path("benchmarks/leetcode/tasks")


def _resolve_langs(value: str | None) -> tuple[str, ...]:
    if not value or value in ("all", "ALL"):
        return LANGUAGES
    langs = tuple(tok.strip() for tok in value.split(",") if tok.strip())
    bad = [tok for tok in langs if tok not in LANGUAGES]
    if bad:
        raise SystemExit(f"Unknown language(s): {', '.join(bad)} (want one of {', '.join(LANGUAGES)} or 'all')")
    return langs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Harbor leetcode tasks.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help=f"Path to the multilingual JSONL dataset (default: {DEFAULT_DATASET}).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output directory for generated tasks (default: {DEFAULT_OUTPUT}).")
    parser.add_argument("--lang", default="all",
                        help="Comma-separated languages or 'all' (default: all).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Generate only the first N problems (each in every language).")
    parser.add_argument("--question-id", type=int, action="append",
                        help="Generate only this question ID (repeatable).")
    parser.add_argument("--with-oracle", action="store_true",
                        help="Also emit a reference solution/solution.json (needs Doocs checkout).")
    parser.add_argument("--skip-unsupported", action="store_true",
                        help="Record unsupported transports in exclusions.json instead of aborting.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing leetcode-<language>.sif images. "
            "Defaults to $LEETCODE_IMAGE_DIR when set."
        ),
    )
    parser.add_argument("--image-python", help="Prebuilt Singularity .sif for python tasks.")
    parser.add_argument("--image-cpp", help="Prebuilt Singularity .sif for cpp tasks.")
    parser.add_argument("--image-go", help="Prebuilt Singularity .sif for go tasks.")
    parser.add_argument("--image-java", help="Prebuilt Singularity .sif for java tasks.")
    parser.add_argument("--image-rust", help="Prebuilt Singularity .sif for rust tasks.")
    parser.add_argument("--image-javascript", help="Prebuilt Singularity .sif for javascript tasks.")
    parser.add_argument("--image-typescript", help="Prebuilt Singularity .sif for typescript tasks.")
    parser.add_argument("--image-php", help="Prebuilt Singularity .sif for php tasks.")
    parser.add_argument("--image-ruby", help="Prebuilt Singularity .sif for ruby tasks.")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    languages = _resolve_langs(args.lang)

    problems = load_problems(args.dataset)
    if args.question_id:
        wanted = set(args.question_id)
        problems = [p for p in problems if p["question_id"] in wanted]
        found = {p["question_id"] for p in problems}
        if found != wanted:
            missing = sorted(wanted - found)
            raise SystemExit(f"Question ID(s) absent from dataset: {missing}")

    if args.limit is not None and args.limit >= 0:
        problems = problems[: args.limit]

    if args.output_dir.exists():
        # Keep compatible: generator requires a fresh dir (previous behavior).
        raise SystemExit(f"Output directory already exists: {args.output_dir}. Use a fresh --output-dir.")

    image_dir = args.image_dir
    if image_dir is None:
        env_image_dir = os.environ.get("LEETCODE_IMAGE_DIR")
        if env_image_dir:
            image_dir = Path(env_image_dir)

    images: dict[str, str] = {}

    for lang in LANGUAGES:
        explicit = getattr(args, f"image_{lang}")

        if explicit:
            images[lang] = explicit
            continue

        if image_dir is not None:
            image = image_dir / f"leetcode-{lang}.sif"

            if not image.is_file():
                raise SystemExit(
                    f"Missing image for {lang}: {image}"
                )

            images[lang] = str(image)

    exclusions, count = generate_all(
        problems,
        output=args.output_dir,
        languages=languages,
        images=images,
        with_oracle=args.with_oracle,
        skip_unsupported=args.skip_unsupported,
    )
    print(f"Generated {count} tasks in {args.output_dir}; excluded {len(exclusions)}")


if __name__ == "__main__":
    main()