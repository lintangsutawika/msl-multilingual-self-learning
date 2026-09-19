from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[3]

DOOCS = ROOT / "external/doocs-leetcode"

CACHE = (
    ROOT
    / "benchmarks"
    / "leetcode"
    / "data"
    / "cache"
)

SNIPPET_DIR = CACHE / "leetcode_snippets"

PROBLEM_LIST_CACHE = CACHE / "leetcode_problem_list.json"

FAILURES = CACHE / "leetcode_snippet_fetch_failures.json"

HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://leetcode.com/problemset/",
}

QUERY = r"""
query questionEditorData($titleSlug: String!) {
  question(titleSlug: $titleSlug) {
    questionId
    questionFrontendId
    title
    titleSlug
    codeSnippets {
      lang
      langSlug
      code
    }
  }
}
"""


def fetch_question(slug: str) -> dict:
    last_error: Exception | None = None

    for attempt in range(5):
        try:
            response = requests.post(
                "https://leetcode.com/graphql/",
                json={
                    "query": QUERY,
                    "variables": {
                        "titleSlug": slug,
                    },
                },
                headers={
                    **HEADERS,
                    "Referer": (
                        "https://leetcode.com/"
                        f"problems/{slug}/"
                    ),
                },
                timeout=30,
            )

            response.raise_for_status()

            data = response.json()

            if data.get("errors"):
                raise RuntimeError(data["errors"])

            question = (
                data
                .get("data", {})
                .get("question")
            )

            if not question:
                raise RuntimeError(
                    "question returned null"
                )

            return question

        except Exception as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))

    assert last_error is not None
    raise last_error


def main() -> None:
    SNIPPET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    doocs_ids: set[int] = set()

    for path in DOOCS.rglob("*"):
        if not path.is_dir():
            continue

        match = re.match(
            r"^(\d+)\.",
            path.name,
        )

        if match:
            doocs_ids.add(
                int(match.group(1))
            )

    if not doocs_ids:
        raise RuntimeError(
            f"No Doocs problem IDs found under {DOOCS}"
        )

    print(
        "Doocs IDs:",
        len(doocs_ids),
    )

    response = requests.get(
        "https://leetcode.com/api/problems/all/",
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    pairs = response.json()["stat_status_pairs"]

    problem_list = []
    by_id: dict[int, dict] = {}

    for item in pairs:
        stat = item["stat"]

        qid = stat.get("frontend_question_id")
        slug = stat.get("question__title_slug")
        title = stat.get("question__title")

        if qid is None or not slug:
            continue

        try:
            qid = int(qid)
        except (TypeError, ValueError):
            continue

        record = {
            "questionFrontendId": str(qid),
            "title": title,
            "titleSlug": slug,
        }

        problem_list.append(record)
        by_id[qid] = record

    PROBLEM_LIST_CACHE.write_text(
        json.dumps(
            problem_list,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    targets = [
        qid
        for qid in sorted(doocs_ids)
        if qid in by_id
    ]

    cached = 0
    fetched = 0
    failed = []

    for i, qid in enumerate(
        targets,
        start=1,
    ):
        out = SNIPPET_DIR / f"{qid}.json"

        if out.exists():
            try:
                existing = json.loads(
                    out.read_text()
                )

                if "code_snippets" in existing:
                    cached += 1
                    continue

            except Exception:
                pass

        slug = by_id[qid]["titleSlug"]

        try:
            question = fetch_question(slug)

            record = {
                "question_id": qid,
                "title": question["title"],
                "title_slug": question["titleSlug"],
                "code_snippets": (
                    question.get("codeSnippets")
                    or []
                ),
            }

            out.write_text(
                json.dumps(
                    record,
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            fetched += 1

        except Exception as exc:
            failed.append(
                {
                    "question_id": qid,
                    "title_slug": slug,
                    "error": str(exc),
                }
            )

            print(
                f"FAILED q{qid}: {exc}"
            )

        if i % 50 == 0:
            print(
                f"[{i}/{len(targets)}] "
                f"cached={cached} "
                f"fetched={fetched} "
                f"failed={len(failed)}"
            )

        time.sleep(0.08)

    FAILURES.write_text(
        json.dumps(
            failed,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("DONE")
    print("targets :", len(targets))
    print("cached  :", cached)
    print("fetched :", fetched)
    print("failed  :", len(failed))


if __name__ == "__main__":
    main()
