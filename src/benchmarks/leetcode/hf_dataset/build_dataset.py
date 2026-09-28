"""Build the neulab/leetcode Hugging Face dataset from LeetCode + GraphQL.

This is the PREP step (build once, push to HF). It fetches raw LeetCode rows from
newfacade/LeetCodeDataset, pulls per-language codeSnippets via LeetCode GraphQL,
constructs the 9-language `interfaces` + `canonical_tests` per problem, flattens to
one row per (problem, language), and writes `data/<split>/<lang>-NNN.jsonl`
(row-boundary shards so every file stays under HF's ~10 MiB per-file ceiling; HF
globs them into one split). After `git push`,
`datasets.load_dataset("neulab/leetcode", split=...)`
returns the rows, which dataset.py pulls at generation time.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root

from src.benchmarks.leetcode.adapter import LANGUAGES  # noqa: E402


"""Build LeetCode problem rows (the execution-dataset schema) live from HF.

Replaces the old committed ``leetcode_multilingual_*.jsonl`` + ``generate_dataset.py``:
``load_problems_hf(split)`` fetches ``newfacade/LeetCodeDataset`` rows for a split and
reconstructs the per-language ``interfaces`` (fetched lazily from LeetCode's
``codeSnippets`` API and cached locally) and ``canonical_tests`` (from HF ``test``),
returning the same problem-row dicts that ``main.py``/``adapter`` consume. No committed
JSONL, no doocs checkout, no manual cache prep: snippets are fetched on demand and
cached under ``benchmarks/leetcode/data/cache/leetcode_snippets``.
"""

import json
import re

from dataclasses import dataclass
from datasets import load_dataset

import requests
import time
from pathlib import Path
from typing import Any

from src.benchmarks.leetcode.hf_dataset.derive import parse_hf_python_signature
from src.benchmarks.leetcode.hf_dataset.parsers import (
    LANGUAGE_SLUGS,
    parse_leetcode_interface,
)


LANGUAGES = (
    "python",
    "cpp",
    "rust",
    "javascript",
    "typescript",
    "go",
    "java",
    "php",
    "ruby",
)

DATASET_NAME = "newfacade/LeetCodeDataset"
# Our hosted, post-interface dataset on HF (pushed from benchmarks/leetcode/hf_dataset).
NEULAB_HF_DATASET = "neulab/leetcode"

LEETCODE_SNIPPET_CACHE = Path(
    "benchmarks/leetcode/data/cache/leetcode_snippets"
)

DEFAULT_OUTPUT = Path(
    "benchmarks/leetcode/data/leetcode_multilingual.jsonl"
)


def _problem_value(
    problem: Any,
    name: str,
    default: Any = None,
) -> Any:
    return getattr(
        problem,
        name,
        default,
    )


def _normalize_whitespace(
    text: str,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _normalize_cpp(
    text: str,
) -> str:
    text = _normalize_whitespace(
        text
    )

    text = re.sub(
        r"\s*&\s*",
        "&",
        text,
    )

    text = re.sub(
        r"\s*\*\s*",
        "*",
        text,
    )

    return text


def _parameter_type_appears_in_signature(
    language: str,
    raw_signature: str,
    name: str,
    type_name: str,
) -> bool:
    if language == "python":
        normalized = _normalize_whitespace(
            raw_signature
        )

        pattern = re.compile(
            rf"\b{re.escape(name)}\s*:\s*"
            rf"{re.escape(type_name)}"
        )

        return bool(
            pattern.search(
                normalized
            )
        )

    if language == "cpp":
        signature = _normalize_cpp(
            raw_signature
        )

        fragment = _normalize_cpp(
            f"{type_name} {name}"
        )

        return fragment in signature

    if language == "go":
        signature = _normalize_whitespace(
            raw_signature
        )

        fragment = f"{name} {type_name}"

        return fragment in signature

    if language == "java":
        signature = _normalize_whitespace(
            raw_signature
        )

        fragment = f"{type_name} {name}"

        return fragment in signature

    raise ValueError(
        f"Unsupported language: {language}"
    )


def _go_return_type_appears(
    raw_signature: str,
    return_type: str,
) -> bool:
    signature = _normalize_whitespace(
        raw_signature
    )

    # Plain return:
    # func foo(x int) []int {
    plain_pattern = re.compile(
        rf"\)\s*"
        rf"{re.escape(return_type)}"
        rf"\s*\{{"
    )

    if plain_pattern.search(
        signature
    ):
        return True

    # Named return:
    # func foo(x int) (ans []int) {
    named_pattern = re.compile(
        rf"\)\s*\([^)]*\s"
        rf"{re.escape(return_type)}"
        rf"\s*\)\s*\{{"
    )

    return bool(
        named_pattern.search(
            signature
        )
    )


def _return_type_appears_in_signature(
    language: str,
    raw_signature: str,
    callable_name: str,
    return_type: str,
) -> bool:
    if language == "python":
        normalized = _normalize_whitespace(
            raw_signature
        )

        return (
            f"-> {return_type}"
            in normalized
        )

    if language == "cpp":
        signature = _normalize_cpp(
            raw_signature
        )

        fragment = _normalize_cpp(
            f"{return_type} {callable_name}"
        )

        return fragment in signature

    if language == "go":
        return _go_return_type_appears(
            raw_signature,
            return_type,
        )

    if language == "java":
        signature = _normalize_whitespace(
            raw_signature
        )

        fragment = (
            f"{return_type} {callable_name}"
        )

        return fragment in signature

    raise ValueError(
        f"Unsupported language: {language}"
    )


def _interface_to_dict(
    interface: Any,
) -> dict[str, Any]:
    return {
        "raw_signature": interface.raw_signature,
        "container": interface.container,
        "callable": interface.callable_name,
        "parameters": [
            {
                "name": parameter.name,
                "type": parameter.type,
            }
            for parameter in interface.parameters
        ],
        "return_type": interface.return_type,
        "signature_source": "leetcode/codeSnippets",
        "type_source": "leetcode/codeSnippets",
    }

_GRAPHQL_URL = "https://leetcode.com/graphql"
_GRAPHQL_HEADERS = {
    "Content-Type": "application/json",
    "Referer": "https://leetcode.com/problems/",
}


# One-time train/full builds hit LeetCode's GraphQL repeatedly; pace requests and
# back off on rate-limits so we don't get dropped by transient 429s/5xx. Resume is
# provided by the per-question snippet cache: a re-run skips already-cached IDs.
_RATE_DELAY = 3.0          # seconds between GraphQL requests (set via --rate-delay)
_SNIP_MAX_RETRIES = 5      # backoff attempts before giving up on one question
_SNIP_BASE_BACKOFF = 2.0   # first retry backoff (seconds); doubles each attempt


def _paced_via_cache(question_id: int) -> dict[str, str] | None:
    """Return cached snippets for question_id, or None if not cached yet."""
    cached = LEETCODE_SNIPPET_CACHE / f"{question_id}.json"
    if cached.is_file():
        try:
            rec = json.loads(cached.read_text("utf-8"))
            return {sn["langSlug"]: sn["code"] for sn in rec.get("code_snippets", [])}
        except (OSError, ValueError):
            pass  # corrupt cache entry -> re-fetch
    return None


def _fetch_snippets(question_id: int, title_slug: str) -> dict[str, str] | None:
    """Fetch a problem's per-language codeSnippets from LeetCode with rate-limit
    + backoff, and cache them.

    Returns {langSlug: code} on success, or None after exhausting retries (the
    caller records a skip). Caches the raw snippet record under
    leetcode_snippets/<qid>.json so repeat builds don't re-hit the API and failed
    runs resume from where they stopped.
    """
    cached = _paced_via_cache(question_id)
    if cached is not None:
        return cached

    query = {
        "query": (
            "query questionData($titleSlug: String!) { "
            "question(titleSlug: $titleSlug) { "
            "questionId title codeSnippets { langSlug code } } }"
        ),
        "variables": {"titleSlug": title_slug},
    }
    snippets = None
    for attempt in range(_SNIP_MAX_RETRIES):
        try:
            resp = requests.post(
                _GRAPHQL_URL,
                json=query,
                headers=_GRAPHQL_HEADERS,
                timeout=30,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = float(resp.headers.get("Retry-After", 0) or 0)
                backoff = retry_after or _SNIP_BASE_BACKOFF * (2 ** attempt)
                print(f"    rate-limited (HTTP {resp.status_code}); "
                      f"backing off {backoff:.0f}s")
                time.sleep(backoff)
                continue
            resp.raise_for_status()
            qn = (resp.json().get("data") or {}).get("question") or {}
            snippets = qn.get("codeSnippets")
            break
        except requests.RequestException:
            # transient network error: brief backoff then retry
            time.sleep(_SNIP_BASE_BACKOFF * (2 ** attempt))
            continue

    if not snippets:
        return None

    record = {"question_id": question_id, "code_snippets": snippets}
    out = LEETCODE_SNIPPET_CACHE / f"{question_id}.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass  # cache is best-effort; interfaces still usable this run

    time.sleep(_RATE_DELAY)  # pace between successful fetches
    return {sn["langSlug"]: sn["code"] for sn in snippets}


def _resolve_all_interfaces(
    problem: Any,
) -> dict[str, Any] | None:
    question_id = int(
        _problem_value(
            problem,
            "question_id",
        )
    )

    cache_path = (
        LEETCODE_SNIPPET_CACHE
        / f"{question_id}.json"
    )

    if not cache_path.exists():
        title_slug = _problem_value(problem, "task_id", "")
        snippets = _fetch_snippets(question_id, title_slug)
        if snippets is None:
            return None
    else:
        record = json.loads(
            cache_path.read_text(
                encoding="utf-8",
            )
        )
        snippets = {
            snippet["langSlug"]: snippet["code"]
            for snippet in record["code_snippets"]
        }

    interfaces: dict[str, Any] = {}

    for language in LANGUAGES:
        slug = LANGUAGE_SLUGS[language]

        code = snippets.get(slug)

        if not code:
            return None

        try:
            interface = parse_leetcode_interface(
                question_id,
                language,
                code,
            )
        except Exception:
            # One malformed language signature (e.g. an unusual C++/PHP param form
            # the parser doesn't handle) should skip the whole problem, not crash
            # the dataset build -- mirrors the old generate_dataset skip behavior.
            return None

        if interface is None:
            return None

        interfaces[language] = (
            _interface_to_dict(
                interface,
            )
        )

    return interfaces

def _build_record(
    problem: Any,
    interfaces: dict[str, Any],
) -> dict[str, Any]:
    return {
        "question_id": _problem_value(
            problem,
            "question_id",
        ),
        "task_id": _problem_value(
            problem,
            "task_id",
        ),
        "difficulty": _problem_value(
            problem,
            "difficulty",
        ),
        "problem_description": _problem_value(
            problem,
            "problem_description",
        ),
        "interfaces": interfaces,
        "canonical_tests": {
            "language": "python",
            "source": _problem_value(
                problem,
                "test",
            ),
        },
        "metadata": {
            "canonical_parameter_names": [
                parameter.name
                for parameter in parse_hf_python_signature(
                    _problem_value(problem, "starter_code")
                ).parameters
            ],
            "leetcode_dataset_entry_point": (
                _problem_value(
                    problem,
                    "entry_point",
                )
            ),
            "problem_source": (
                "newfacade/LeetCodeDataset"
            ),
            "interface_source": (
                "leetcode/codeSnippets"
            ),
        },
    }




SPLITS = ("train", "test")


def build_problems_hf(
    split: str = "test",
    *,
    rate_delay: float = 3.0,
    limit: int | None = None,
) -> "list[dict]":
    """Build post-interface rows for a split via HF newfacade + LeetCode GraphQL.

    This is the PREP step: it fetches codeSnippets + interfaces and returns the
    (nested) per-problem records, then flatten + JSONL happens downstream. Requests
    are throttled (--rate-delay) with backoff so one-time train/full builds don't
    trip LeetCode rate limits; the per-question snippet cache makes re-runs resume.
    `limit` caps how many problems are processed (for smoke-testing a subset).
    """
    global _RATE_DELAY
    _RATE_DELAY = rate_delay

    if split == "train":
        problems = load_train_split()
    else:
        problems = load_test_split()
    if limit is not None:
        problems = problems[:limit]

    records: list[dict] = []
    skipped = []
    total = len(problems)
    for idx, problem in enumerate(problems, start=1):
        qid = _problem_value(problem, "question_id")
        print(f"[{split}] {idx}/{total} qid={qid}", flush=True)
        interfaces = _resolve_all_interfaces(problem)
        if interfaces is None:
            skipped.append((qid, _problem_value(problem, "task_id")))
            continue
        records.append(_build_record(problem, interfaces))
    if not records:
        raise RuntimeError("No problems with complete LeetCode interfaces were generated")
    if skipped:
        print(f"[{split}] skipped {len(skipped)} (no complete interface): {skipped}")
    return records


def flatten(records):
    """Expand nested problem-records into one row per (problem, language)."""
    flat = []
    for rec in records:
        interfaces = rec.get("interfaces")
        if not interfaces:
            flat.append(rec)
            continue
        base = {k: v for k, v in rec.items() if k != "interfaces"}
        for language, interface in interfaces.items():
            row = dict(base)
            row["language"] = language
            row["interface"] = interface
            flat.append(row)
    return flat


def _clean_json_line(r) -> str:
    """Serialize a row to a single JSONL line, escaping raw line-separator chars
    (U+0085/U+2028/U+2029/CR) that HF's line reader would otherwise split on."""
    line = __import__("json").dumps(r, ensure_ascii=False)
    for ch, esc in (("\x85", "\\u0085"),
                    ("\u2028", "\\u2028"),
                    ("\u2029", "\\u2029"),
                    ("\r", "\\r")):
        line = line.replace(ch, esc)
    return line + "\n"


def build(split, out, *, rate_delay=3.0, limit=None, shard_mb=8):
    records = build_problems_hf(split, rate_delay=rate_delay, limit=limit)
    if not records:
        return
    flat = flatten(records)
    split_dir = out / "data" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    by_lang = {}
    for row in flat:
        by_lang.setdefault(row.get("language"), []).append(row)
    total = 0
    shard_bytes = shard_mb * 1024 * 1024
    for lang, rows in sorted(by_lang.items()):
        # Write rows as row-boundary shards so every file stays well under HF's
        # ~10 MiB per-file ceiling; HF globs data/<split>/*.jsonl into one split.
        shard_idx = 0
        fh = None
        used = 0
        n_shard = 0
        for r in rows:
            line = _clean_json_line(r)
            if fh is None or (used > 0 and used + len(line) > shard_bytes):
                if fh is not None:
                    fh.close()
                fh = (split_dir / f"{lang}-{shard_idx:03d}.jsonl").open("w", encoding="utf-8")
                shard_idx += 1
                used = 0
                n_shard += 1
            fh.write(line)
            used += len(line)
        if fh is not None:
            fh.close()
        total += len(rows)
        files = sorted(split_dir.glob(f"{lang}-*.jsonl"))
        print(f"[build] {split}/{lang}: {len(rows)} rows "
              f"(sharded {n_shard}: {', '.join(f.name for f in files)})")
    print(f"[build] {split}: {len(flat)} problems -> {total} rows total")


def write_readme(out):
    readme = """---
license: apache-2.0
language:
- code
task_categories:
- text-generation
---

# LeetCode multilingual benchmark dataset

Post-interface rows for the msl-multilingual-self-learning benchmark, flattened to
one row per (problem, language), stored as `data/<split>/<lang>-NNN.jsonl`
(row-boundary shards so every file stays under HF's ~10 MiB per-file ceiling; HF
globs them into a single split). Each row has the `interface` for its `language`,
plus the shared
`canonical_tests` oracle, `problem_description`, and `metadata`.

## Load

```python
from datasets import load_dataset
train = load_dataset("neulab/leetcode", split="train")
test  = load_dataset("neulab/leetcode", split="test")
```
"""


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Prep a LeetCode split into per-language JSONL for the HF "
                    "dataset push (throttled GraphQL fetch; resumable via cache)."
    )
    ap.add_argument("--out", type=Path, default=Path("/home/aci18914wh/leetcode"))
    ap.add_argument("--only", choices=SPLITS, default=None,
                    help="Build only this split (default: all splits).")
    ap.add_argument("--rate-delay", type=float, default=3.0,
                    help="Seconds between GraphQL requests (default 3.0).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N problems (smoke-test a subset).")
    ap.add_argument("--shard-mb", type=float, default=8.0,
                    help="Target max MiB per output file (default 8, under HF's ~10 "
                         "MiB ceiling); rows are sliced on boundaries.")
    args = ap.parse_args()
    splits = (args.only,) if args.only else SPLITS
    for split in splits:
        build(split, args.out, rate_delay=args.rate_delay, limit=args.limit,
              shard_mb=args.shard_mb)
    write_readme(args.out)


# --- raw LeetCode row loading (prep) ----------------------------------------
DATASET_NAME = "newfacade/LeetCodeDataset"


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


def _rows_to_problems(dataset) -> list[LeetCodeProblem]:
    problems: list[LeetCodeProblem] = []
    for row in dataset:
        problems.append(LeetCodeProblem(
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
        ))
    return problems


def load_train_split() -> list[LeetCodeProblem]:
    return _rows_to_problems(load_dataset(DATASET_NAME, split="train"))


def load_test_split() -> list[LeetCodeProblem]:
    return _rows_to_problems(load_dataset(DATASET_NAME, split="test"))


if __name__ == "__main__":
    main()
