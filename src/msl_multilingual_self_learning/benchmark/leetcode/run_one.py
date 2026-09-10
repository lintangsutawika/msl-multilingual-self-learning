from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def format_entrypoint_requirement(language: str, interface: dict[str, Any]) -> str:
    return (
        f"Entrypoint: {interface['raw_signature'].strip()}\n"
        f"Container: {interface.get('container') or '(none)'}"
    )


def build_prompt(record: dict[str, Any], language: str) -> str:
    """One template for all languages; no starter code, examples of code, or tests."""
    interface = record["interfaces"][language]
    return (
        f"Language: {language}\n"
        f"{format_entrypoint_requirement(language, interface)}\n\n"
        f"Problem:\n{record['problem_description'].strip()}\n\n"
        "Solve the problem in the specified language using the required entrypoint. "
        "Return a complete source file, including any required imports, includes, "
        "and package declarations. Do not define a program entrypoint. "
        "The source must compile or run as submitted. Do not include unused imports. "
        "Return only source code, without Markdown fences or explanations."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load one LeetCode benchmark "
            "problem and construct its "
            "model prompt."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--question-id",
        type=int,
        default=3243,
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
