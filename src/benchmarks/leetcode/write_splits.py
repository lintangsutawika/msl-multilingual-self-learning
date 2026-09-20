from __future__ import annotations

import json
from pathlib import Path

from .selection import (
    ALL_9_LANGUAGES,
    MAIN_4_LANGUAGES,
    build_splits,
    load_leetcode_snippet_records,
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
        "a1": {
            "description": (
                "LeetCode problems supporting all 9 target languages"
            ),
            "languages": sorted(ALL_9_LANGUAGES),
        },
        "a2": {
            "description": (
                "LeetCode problems supporting Python3, C++, Go, and Java"
            ),
            "languages": sorted(MAIN_4_LANGUAGES),
        },
        "b": {
            "description": (
                "A1 intersected with newfacade/LeetCodeDataset test split"
            ),
            "languages": sorted(ALL_9_LANGUAGES),
        },
    }

    for name in ("a1", "a2", "b"):
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
