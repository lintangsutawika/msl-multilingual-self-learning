from __future__ import annotations

import json
from pathlib import Path

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


def load_hf_train_ids() -> set[int]:
    dataset = load_dataset(
        HF_DATASET_NAME,
        split="train",
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

    # Test = the a1 problems that newfacade's dataset marks as test (held out).
    test = a1 & hf_test_ids
    # Train = the a1 universe MINUS every newfacade-test problem, so train and test
    # (and train vs the whole newfacade test set) have zero overlap.
    train = a1 - hf_test_ids

    return {
        "train": train,
        "test": test,
    }



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

OUTPUT_DIR = Path(
    "benchmarks/leetcode/data/splits"
)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    splits = build_splits()
    records = load_leetcode_snippet_records()

    metadata = {
        "train": {
            "description": (
                "LeetCode problems supporting all 9 target languages (the train pool)"
            ),
            "languages": sorted(ALL_9_LANGUAGES),
        },
        "test": {
            "description": (
                "Train intersected with newfacade/LeetCodeDataset test split "
                "(the held-out eval set)"
            ),
            "languages": sorted(ALL_9_LANGUAGES),
        },
    }

    for name in ("train", "test"):
        ids = sorted(splits[name])

        questions = []

        for qid in ids:
            record = records[qid]

            questions.append(
                {
                    "question_id": qid,
                    "title": record["title"],
                    "title_slug": record["title_slug"],
                    "available_languages": sorted(
                        {
                            snippet["langSlug"]
                            for snippet
                            in record["code_snippets"]
                        }
                    ),
                }
            )

        payload = {
            "name": name,
            "count": len(ids),
            **metadata[name],
            "questions": questions,
        }

        out = OUTPUT_DIR / f"{name}.json"

        out.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            f"{name.upper():<3} "
            f"{len(ids):>4} "
            f"-> {out}"
        )


if __name__ == "__main__":
    main()
