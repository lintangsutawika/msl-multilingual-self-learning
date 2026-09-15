from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_NAME = "newfacade/LeetCodeDataset"
DEFAULT_DATASET = Path("benchmarks/leetcode/data/leetcode_multilingual.jsonl")


@dataclass
class LeetCodeProblem:
    task_id: str
    question_id: int
    difficulty: str
    tags: list[str]

    problem_description: str
    starter_code: str
    entry_point: str
    test: str
    completion: str

    input_output: list[dict[str, str]]


def load_test_split() -> list[LeetCodeProblem]:
    dataset = load_dataset(
        DATASET_NAME,
        split="test",
    )

    problems: list[LeetCodeProblem] = []

    for row in dataset:
        problem = LeetCodeProblem(
            task_id=row["task_id"],
            question_id=row["question_id"],
            difficulty=row["difficulty"],
            tags=row["tags"],
            problem_description=row["problem_description"],
            starter_code=row["starter_code"],
            entry_point=row["entry_point"],
            test=row["test"],
            input_output=row["input_output"],
            completion=row["completion"],
        )

        problems.append(problem)

    return problems


def load_problem(
    dataset_path: Path,
    question_id: int,
) -> dict[str, Any]:
    """Look up one generated JSONL record (the CLI/adapter input) by question_id."""
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