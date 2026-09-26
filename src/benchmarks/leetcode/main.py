"""Generate Harbor LeetCode tasks.

Examples:
    # Prepared benchmark sets
    uv run python -m src.benchmarks.leetcode --set a1
    uv run python -m src.benchmarks.leetcode --set a2
    uv run python -m src.benchmarks.leetcode --set b

    # Ad-hoc generation
    uv run python -m src.benchmarks.leetcode --limit 10 --lang all
    uv run python -m src.benchmarks.leetcode --lang python --question-id 3243
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .adapter import LANGUAGES, generate_all


DEFAULT_DATASET = Path(
    "benchmarks/leetcode/data/leetcode_multilingual_leetcode.jsonl"
)
DEFAULT_OUTPUT = Path("benchmarks/leetcode/tasks")
SPLIT_DIR = Path("benchmarks/leetcode/data/splits")

MAIN_4_LANGUAGES = (
    "python",
    "cpp",
    "go",
    "java",
)

SET_LANGUAGES = {
    "a1": LANGUAGES,
    "a2": MAIN_4_LANGUAGES,
    "b": LANGUAGES,
}


def load_problems(dataset: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in dataset.read_text().splitlines()
        if line.strip()
    ]


def load_split_ids(name: str) -> set[int]:
    path = SPLIT_DIR / f"{name}.json"

    if not path.is_file():
        raise SystemExit(
            f"Split manifest not found: {path}. "
            "Run: uv run python -m src.benchmarks.leetcode.write_splits"
        )

    payload = json.loads(path.read_text())

    return {
        int(question["question_id"])
        for question in payload["questions"]
    }


def _resolve_langs(value: str | None) -> tuple[str, ...]:
    if not value or value in ("all", "ALL"):
        return LANGUAGES

    langs = tuple(
        tok.strip()
        for tok in value.split(",")
        if tok.strip()
    )

    bad = [
        tok
        for tok in langs
        if tok not in LANGUAGES
    ]

    if bad:
        raise SystemExit(
            "Unknown language(s): "
            f"{', '.join(bad)} "
            f"(want one of {', '.join(LANGUAGES)} or 'all')"
        )

    return langs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Harbor LeetCode tasks."
    )

    parser.add_argument(
        "--set",
        choices=("a1", "a2", "b"),
        default=None,
        help=(
            "Prepared benchmark set: "
            "a1=full 9-language universe, "
            "a2=full 4-language universe, "
            "b=LeetCodeDataset test split in all 9 languages."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=(
            "Path to the multilingual execution JSONL "
            f"(default: {DEFAULT_DATASET})."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to benchmarks/leetcode/tasks-<set> "
            "when --set is used, otherwise benchmarks/leetcode/tasks."
        ),
    )

    parser.add_argument(
        "--lang",
        default=None,
        help=(
            "Comma-separated languages or 'all'. "
            "When --set is used, defaults to that set's language list."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Generate only the first N selected problems.",
    )

    parser.add_argument(
        "--question-id",
        type=int,
        action="append",
        help="Generate only this question ID (repeatable).",
    )

    parser.add_argument(
        "--with-oracle",
        action="store_true",
        help=(
            "Also emit a reference solution/solution.json "
            "(needs Doocs checkout)."
        ),
    )

    parser.add_argument(
        "--skip-unsupported",
        action="store_true",
        help=(
            "Record unsupported transports in exclusions.json "
            "instead of aborting."
        ),
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing leetcode-<language>.sif images. "
            "Defaults to $LEETCODE_IMAGE_DIR when set."
        ),
    )

    for lang in LANGUAGES:
        parser.add_argument(
            f"--image-{lang}",
            help=f"Prebuilt Singularity .sif for {lang} tasks.",
        )

    parser.add_argument(
        "--prebuild-sif",
        action="store_true",
        help=(
            "Build a Singularity sif from each language template's "
            "environment/Dockerfile into the output dir and set it as "
            "task.toml [environment].docker_image (overrides --image-dir)."
        ),
    )
    parser.add_argument(
        "--prebuild-dir",
        type=Path,
        default=None,
        help="Where prebuilt sifs are written (default: the task output dir).",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.set is not None:
        default_languages = SET_LANGUAGES[args.set]

        if args.lang is None:
            languages = tuple(default_languages)
        else:
            languages = _resolve_langs(args.lang)

        if args.output_dir is None:
            output_dir = Path(
                f"benchmarks/leetcode/tasks-{args.set}"
            )
        else:
            output_dir = args.output_dir
    else:
        languages = _resolve_langs(args.lang)
        output_dir = args.output_dir or DEFAULT_OUTPUT

    problems = load_problems(args.dataset)

    if args.set is not None:
        wanted = load_split_ids(args.set)

        available = {
            int(problem["question_id"])
            for problem in problems
        }

        missing = sorted(wanted - available)

        if missing:
            preview = ", ".join(
                str(qid)
                for qid in missing[:10]
            )

            raise SystemExit(
                f"Dataset {args.dataset} does not contain "
                f"{len(missing)} question(s) required by set {args.set}. "
                f"First missing IDs: {preview}. "
                "Generate/use the corresponding execution dataset first."
            )

        problems = [
            problem
            for problem in problems
            if int(problem["question_id"]) in wanted
        ]

    if args.question_id:
        wanted = set(args.question_id)

        problems = [
            problem
            for problem in problems
            if int(problem["question_id"]) in wanted
        ]

        found = {
            int(problem["question_id"])
            for problem in problems
        }

        if found != wanted:
            missing = sorted(wanted - found)
            raise SystemExit(
                f"Question ID(s) absent from selected dataset/set: {missing}"
            )

    problems.sort(
        key=lambda problem: int(problem["question_id"])
    )

    if args.limit is not None and args.limit >= 0:
        problems = problems[: args.limit]

    if output_dir.exists():
        raise SystemExit(
            f"Output directory already exists: {output_dir}. "
            "Use a fresh --output-dir or remove the generated directory."
        )

    image_dir = args.image_dir

    if image_dir is None:
        env_image_dir = os.environ.get(
            "LEETCODE_IMAGE_DIR"
        )

        if env_image_dir:
            image_dir = Path(env_image_dir)

    images: dict[str, str] = {}

    for lang in LANGUAGES:
        explicit = getattr(
            args,
            f"image_{lang}",
        )

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

    # --prebuild-sif: build one sif per included language from its template
    # Dockerfile, and point task.toml [environment].docker_image at it (saved
    # into the output dir next to the generated tasks).
    if args.prebuild_sif:
        from .adapter import prebuild_language_sifs

        prebuild_dir = args.prebuild_dir or output_dir

        # generate_all atomically renames a staged dir into output_dir, so the
        # sifs cannot live there yet. Build them into a sibling scratch dir,
        # then move them in AFTER generation succeeds.
        scratch = output_dir.parent / f".{output_dir.name}.prebuild"
        scratch.mkdir(parents=True, exist_ok=True)
        try:
            prebuilt_scratch = prebuild_language_sifs(
                languages, scratch,
            )
            # Final sif lives alongside the tasks; task.toml references the
            # absolute final path (so it is correct after the move below).
            final_sifs = {
                lang: str(output_dir / f"{lang}.sif")
                for lang in prebuilt_scratch
            }
            for lang, sif in final_sifs.items():
                images.setdefault(lang, sif)
        except Exception:
            raise

    exclusions, count = generate_all(
        problems,
        output=output_dir,
        languages=languages,
        images=images,
        with_oracle=args.with_oracle,
        skip_unsupported=args.skip_unsupported,
    )

    # Move prebuilt sifs into the task dir now that generation succeeded.
    if args.prebuild_sif:
        scratch = output_dir.parent / f".{output_dir.name}.prebuild"
        for lang in prebuilt_scratch:
            src = scratch / f"{lang}.sif"
            dst = output_dir / f"{lang}.sif"
            if src.exists():
                shutil.move(str(src), str(dst))
        shutil.rmtree(scratch, ignore_errors=True)

    print(
        f"Generated {count} tasks in {output_dir}; "
        f"excluded {len(exclusions)}"
    )


if __name__ == "__main__":
    main()
