"""Fetch and cache the LeetCode problem pages that neulab/leetcode is built from.

One GraphQL request per problem fetches everything the build needs (page HTML,
code snippets, metaData, similar questions, ...). Each response is cached as
<cache-dir>/<question_id>.json, so a build resumes where it stopped and a
rebuild needs no network.
"""
from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from typing import Any

import requests

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "msl-leetcode" / "raw"
GRAPHQL_URL = "https://leetcode.com/graphql"
QUERY = """
query question($titleSlug: String!) {
  question(titleSlug: $titleSlug) {
    questionId
    questionFrontendId
    title
    titleSlug
    difficulty
    categoryTitle
    isPaidOnly
    content
    codeSnippets { langSlug code }
    metaData
    judgeType
    exampleTestcases
    exampleTestcaseList
    topicTags { slug }
    hints
    similarQuestions
    stats
    envInfo
  }
}
"""
MAX_RETRIES = 6
BASE_BACKOFF = 5.0


def fetch(slug: str) -> dict[str, Any] | None:
    """One problem's GraphQL record; None if LeetCode has no such problem."""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                GRAPHQL_URL,
                json={"query": QUERY, "variables": {"titleSlug": slug}},
                headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0",
                         "Referer": f"https://leetcode.com/problems/{slug}/"},
                timeout=30,
            )
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = float(response.headers.get("Retry-After") or 0)
                delay = retry_after or BASE_BACKOFF * 2 ** attempt
                print(f"  HTTP {response.status_code} for {slug}; retrying in {delay:.0f}s", flush=True)
                time.sleep(delay)
                continue
            response.raise_for_status()
            return (response.json().get("data") or {}).get("question")
        except (requests.RequestException, ValueError) as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"  {type(exc).__name__} for {slug}; retrying", flush=True)
            time.sleep(BASE_BACKOFF * 2 ** attempt)
    raise RuntimeError(f"gave up on {slug} after {MAX_RETRIES} attempts")


def load_or_fetch(question_id: int, slug: str, cache_dir: Path = DEFAULT_CACHE_DIR,
                  rate_delay: float = 3.0) -> dict[str, Any] | None:
    """The cached record for a problem, fetching (then pausing rate_delay) if it is not cached.

    None if LeetCode returned nothing for the slug.
    """
    path = cache_dir / f"{question_id}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    question = fetch(slug)
    time.sleep(rate_delay)
    if not question:
        return None
    if str(question.get("questionFrontendId")) != str(question_id):
        print(f"  warning: {slug} has frontend id {question.get('questionFrontendId')}, "
              f"source dataset says {question_id}", flush=True)
    record = {
        "question_id": question_id,
        "slug": slug,
        "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "question": question,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return record
