from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .unified_tasks import build_prompt


DEFAULT_DATASET = Path(
    "benchmarks/leetcode/data/leetcode_multilingual.jsonl"
)


LANGUAGE_NAMES = {
    "python": "Python",
    "cpp": "C++",
    "go": "Go",
    "java": "Java",
}


def load_problem(
    dataset_path: Path,
    question_id: int,
) -> dict[str, Any]:
    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if int(record["question_id"]) == question_id:
                return record

    raise ValueError(
        f"Question ID {question_id} "
        f"not found in {dataset_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the model prompt for one LeetCode "
            "problem/language combination."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=(
            f"Path to the multilingual JSONL dataset. "
            f"Default: {DEFAULT_DATASET}"
        ),
    )
    parser.add_argument(
        "--question-id",
        type=int,
        default=3243,
        help=(
            "LeetCode question ID to render. "
            "Default: 3243"
        ),
    )
    parser.add_argument(
        "--language",
        choices=(
            "python",
            "cpp",
            "go",
            "java",
        ),
        default="python",
        help=(
            "Target language for the prompt. "
            "Default: python"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    record = load_problem(
        args.dataset,
        args.question_id,
    )

    prompt = build_prompt(
        record,
        args.language,
    )

    print(
        "=" * 80
    )

    print(
        f"Question: "
        f"{record['question_id']} "
        f"{record['task_id']}"
    )

    print(
        f"Language: "
        f"{args.language}"
    )

    print(
        "=" * 80
    )

    print()
    print(prompt)


if __name__ == "__main__":
    main()