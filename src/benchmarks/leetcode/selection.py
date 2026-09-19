from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


LEETCODE_SNIPPET_CACHE = Path(
    "benchmarks/leetcode/data/cache/leetcode_snippets"
)

HF_DATASET_NAME = "newfacade/LeetCodeDataset"


ALL_9_LANGUAGES = frozenset(
    {
        "python3",
        "cpp",
        "rust",
        "javascript",
        "typescript",
        "golang",
        "java",
        "php",
        "ruby",
    }
)

MAIN_4_LANGUAGES = frozenset(
    {
        "python3",
        "cpp",
        "golang",
        "java",
    }
)


def load_leetcode_snippet_records(
    cache_dir: Path = LEETCODE_SNIPPET_CACHE,
) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}

    for path in sorted(cache_dir.glob("*.json")):
        record = json.loads(
            path.read_text(encoding="utf-8")
        )

        question_id = int(record["question_id"])
        records[question_id] = record

    return records


def available_languages(
    record: dict[str, Any],
) -> set[str]:
    return {
        snippet["langSlug"]
        for snippet in record["code_snippets"]
    }


def select_questions(
    records: dict[int, dict[str, Any]],
    required_languages: frozenset[str],
) -> set[int]:
    return {
        question_id
        for question_id, record in records.items()
        if required_languages <= available_languages(record)
    }


def load_hf_test_ids() -> set[int]:
    dataset = load_dataset(
        HF_DATASET_NAME,
        split="test",
    )

    return {
        int(row["question_id"])
        for row in dataset
    }


def build_splits() -> dict[str, set[int]]:
    records = load_leetcode_snippet_records()

    a1 = select_questions(
        records,
        ALL_9_LANGUAGES,
    )

    a2 = select_questions(
        records,
        MAIN_4_LANGUAGES,
    )

    hf_test_ids = load_hf_test_ids()

    b1 = a1 & hf_test_ids
    b2 = a2 & hf_test_ids

    return {
        "a1": a1,
        "a2": a2,
        "b1": b1,
        "b2": b2,
    }
