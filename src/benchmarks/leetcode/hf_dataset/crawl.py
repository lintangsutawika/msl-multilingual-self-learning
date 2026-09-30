"""Crawl the LeetCode problem pages that neulab/leetcode is built from.

One GraphQL request per problem fetches everything the build needs: the page
HTML (description with exact exponents, Constraints, images), the per-language
code snippets (interfaces), metaData (LeetCode's typed parameters, e.g.
TreeNode/ListNode), example testcases, tags, hints, the premium flag, related
problems (for train/test overlap checks), acceptance stats and LeetCode's
compiler versions. Each
response is cached as <cache-dir>/<question_id>.json, so a crawl resumes where
it stopped and the build never needs the network.

    uv run python -m src.benchmarks.leetcode.hf_dataset.crawl --limit 5
    uv run python -m src.benchmarks.leetcode.hf_dataset.crawl            # all problems
"""
from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Any

import requests

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "msl-leetcode" / "raw"
SOURCE_DATASET = "newfacade/LeetCodeDataset"
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


def cache_path(cache_dir: Path, question_id: int) -> Path:
    return cache_dir / f"{question_id}.json"


def load_cached(question_id: int, cache_dir: Path = DEFAULT_CACHE_DIR) -> dict[str, Any] | None:
    """The cached crawl record for a problem, or None if it was not crawled."""
    path = cache_path(cache_dir, question_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def source_problems() -> dict[int, str]:
    """{question_id: title slug} for every problem in the source dataset (train + test)."""
    from datasets import load_dataset

    problems: dict[int, str] = {}
    for split in ("train", "test"):
        for row in load_dataset(SOURCE_DATASET, split=split):
            problems[int(row["question_id"])] = row["task_id"]
    return problems


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


def crawl(problems: dict[int, str], cache_dir: Path, *, rate_delay: float, refresh: bool) -> list[int]:
    """Fetch and cache every problem not cached yet; returns the ids LeetCode did not return."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing: list[int] = []
    todo = [(qid, slug) for qid, slug in sorted(problems.items())
            if refresh or not cache_path(cache_dir, qid).is_file()]
    print(f"{len(problems)} problems, {len(problems) - len(todo)} cached, {len(todo)} to fetch", flush=True)
    for n, (qid, slug) in enumerate(todo, 1):
        question = fetch(slug)
        if not question:
            print(f"[{n}/{len(todo)}] {qid} {slug}: not found", flush=True)
            missing.append(qid)
        else:
            if str(question.get("questionFrontendId")) != str(qid):
                print(f"  warning: {slug} has frontend id {question.get('questionFrontendId')}, "
                      f"source dataset says {qid}", flush=True)
            record = {
                "question_id": qid,
                "slug": slug,
                "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "question": question,
            }
            path = cache_path(cache_dir, qid)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
            print(f"[{n}/{len(todo)}] {qid} {slug}", flush=True)
        time.sleep(rate_delay)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--rate-delay", type=float, default=3.0, help="Seconds between requests.")
    parser.add_argument("--limit", type=int, default=None, help="Crawl only the first N problems (by id).")
    parser.add_argument("--ids", type=int, nargs="*", default=None, help="Crawl only these question ids.")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch problems that are already cached.")
    args = parser.parse_args()

    problems = source_problems()
    if args.ids:
        problems = {qid: problems[qid] for qid in args.ids}
    if args.limit is not None:
        problems = dict(sorted(problems.items())[: args.limit])
    missing = crawl(problems, args.cache_dir, rate_delay=args.rate_delay, refresh=args.refresh)
    if missing:
        print(f"LeetCode returned nothing for {len(missing)} problems: {missing}")


if __name__ == "__main__":
    main()
