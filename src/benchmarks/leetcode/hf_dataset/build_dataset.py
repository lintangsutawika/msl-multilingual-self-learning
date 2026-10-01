"""Build the neulab/leetcode Hugging Face dataset from LeetCode + GraphQL.

This is the PREP step (build once, push to HF). It takes the canonical tests from
newfacade/LeetCodeDataset, fetches each problem's LeetCode page (crawl.py), builds
the 9-language `interfaces`, the description and the cleaned tests (clean.py),
flattens to one row per (problem, language), and writes `data/<split>/<lang>-NNN.jsonl`
(row-boundary shards under HF's ~10 MiB per-file ceiling) plus `reports/` listing
every dropped problem and test. After `git push`,
`datasets.load_dataset("neulab/leetcode", split=...)` returns the rows, which
dataset.py pulls at generation time.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

from .clean import (
    apply_comparison,
    apply_drop_categories,
    clean_tests,
    comparison_for,
    drop_list,
    flags,
    mark_overlap,
    problem_drops,
    public_examples,
    public_tests,
    summarize,
)
from .crawl import DEFAULT_CACHE_DIR, load_or_fetch
from .derive import parse_hf_python_signature
from .description import html_to_text
from .parsers import LANGUAGE_SLUGS, parse_leetcode_interface

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
# Our hosted dataset on HF.
NEULAB_HF_DATASET = "neulab/leetcode"


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


def _resolve_all_interfaces(
    problem: LeetCodeProblem,
    question: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Interfaces for the 9 languages, and why any is missing."""
    snippets = {s["langSlug"]: s["code"] for s in question.get("codeSnippets") or []}
    interfaces: dict[str, Any] = {}
    problems: list[str] = []

    for language in LANGUAGES:
        code = snippets.get(LANGUAGE_SLUGS[language])

        if not code:
            problems.append(f"{language}: no code snippet")
            continue

        try:
            interface = parse_leetcode_interface(problem.question_id, language, code)
        except Exception as exc:
            # One malformed signature skips the whole problem (reported).
            problems.append(f"{language}: {type(exc).__name__}: {exc}"[:200])
            continue

        if interface is None:
            problems.append(f"{language}: unparsed signature")
            continue

        interfaces[language] = _interface_to_dict(interface)

    return interfaces, problems


def _build_record(
    problem: LeetCodeProblem,
    question: dict[str, Any],
    crawled: dict[str, Any],
    interfaces: dict[str, Any],
    description: str,
    tests: str,
    public: str,
    entry: dict[str, Any],
) -> dict[str, Any]:
    return {
        "question_id": problem.question_id,
        "task_id": problem.task_id,
        "difficulty": question.get("difficulty") or problem.difficulty,
        "problem_description": description,
        "interfaces": interfaces,
        "canonical_tests": {
            "language": "python",
            "source": apply_comparison(tests, entry["comparison"]),
            "comparison": entry["comparison"],
        },
        # The page's examples that are among the cleaned tests, in the same form.
        "public_tests": {
            "language": "python",
            "source": apply_comparison(public, entry["comparison"]) if public else "",
            "comparison": entry["comparison"],
        },
        "hints": entry["hints"],
        "metadata": {
            "canonical_parameter_names": entry["parameter_names"],
            "leetcode_dataset_entry_point": problem.entry_point,
            "problem_source": DATASET_NAME,
            "description_source": "leetcode.com",
            "interface_source": "leetcode/codeSnippets",
            "fetched_at": crawled["fetched_at"],
            "tests_total": entry["tests"]["total"],
            "tests_dropped": len(entry["dropped_tests"]),
            "flags": entry["flags"],
            "topic_tags": [t["slug"] for t in question.get("topicTags") or []],
        },
    }


def build_problem(
    problem: LeetCodeProblem,
    *,
    cache_dir: Path,
    rate_delay: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """(record or None if dropped, report entry) for one problem."""
    crawled = load_or_fetch(problem.question_id, problem.task_id, cache_dir, rate_delay)
    question = crawled["question"] if crawled else None
    entry: dict[str, Any] = {
        "question_id": problem.question_id,
        "task_id": problem.task_id,
        "drop": problem_drops(question),
    }
    if question is None or entry["drop"] == ["premium"]:
        return None, entry
    try:
        entry["similar"] = [q["titleSlug"] for q in json.loads(question.get("similarQuestions") or "[]")]
    except ValueError:
        entry["similar"] = []

    description = html_to_text(question["content"])
    interfaces, interface_problems = _resolve_all_interfaces(problem, question)
    if interface_problems:
        entry["drop"].append("interface")
        entry["interface_problems"] = interface_problems
    entry["flags"] = flags(problem.question_id, description.text, interfaces, question,
                           len(description.images) + description.videos)
    entry["comparison"] = comparison_for(entry["flags"])
    entry["parameter_names"] = [p.name for p in parse_hf_python_signature(problem.starter_code).parameters]

    examples = public_examples(question, entry["parameter_names"])
    tests, test_report = clean_tests(
        {
            "question_id": problem.question_id,
            "metadata": {"canonical_parameter_names": entry["parameter_names"]},
            "canonical_tests": {"language": "python", "source": problem.test},
            "interfaces": interfaces,
        },
        question["content"],
        examples,
    )
    entry["drop"] += test_report.pop("drop")
    entry.update(test_report)
    if entry["drop"]:
        return None, entry
    public, missing = public_tests(tests, entry["parameter_names"], examples)
    entry["public_tests"] = {"examples": len(examples), "missing": missing}
    entry["hints"] = [html_to_text(h).text.strip() for h in question.get("hints") or []]
    return _build_record(problem, question, crawled, interfaces, description.text, tests, public, entry), entry


SPLITS = ("train", "test")


def build_problems_hf(
    split: str = "test",
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    rate_delay: float = 3.0,
    limit: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Build (records, report entries) for a split; pages are fetched once and cached."""
    if split == "train":
        problems = load_train_split()
    else:
        problems = load_test_split()
    if limit is not None:
        problems = problems[:limit]

    records: list[dict] = []
    report: list[dict] = []
    total = len(problems)
    for idx, problem in enumerate(problems, start=1):
        print(f"[{split}] {idx}/{total} qid={problem.question_id}", flush=True)
        record, entry = build_problem(problem, cache_dir=cache_dir, rate_delay=rate_delay)
        report.append(entry)
        if record is not None:
            records.append(record)
    if not records:
        raise RuntimeError("No problems were kept")
    return records, report


def flatten(records):
    """Expand nested problem-records into one row per (problem, language)."""
    flat = []
    for rec in records:
        base = {k: v for k, v in rec.items() if k != "interfaces"}
        for language, interface in rec["interfaces"].items():
            row = dict(base)
            row["language"] = language
            row["interface"] = interface
            flat.append(row)
    return flat


def _clean_json_line(r) -> str:
    """Serialize a row to a single JSONL line, escaping raw line-separator chars
    (U+0085/U+2028/U+2029/CR) that HF's line reader would otherwise split on."""
    line = json.dumps(r, ensure_ascii=False)
    for ch, esc in (("\x85", "\\u0085"),
                    (" ", "\\u2028"),
                    (" ", "\\u2029"),
                    ("\r", "\\r")):
        line = line.replace(ch, esc)
    return line + "\n"


def build(split, records, out, *, shard_mb=8):
    """Write data/<split>/<lang>-NNN.jsonl, replacing older files."""
    flat = flatten(records)
    split_dir = out / "data" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for old in split_dir.glob("*.jsonl"):
        old.unlink()
    by_lang = {}
    for row in flat:
        by_lang.setdefault(row.get("language"), []).append(row)
    total = 0
    shard_bytes = shard_mb * 1024 * 1024
    for lang, rows in sorted(by_lang.items()):
        shard_idx = 0
        fh = None
        used = 0
        for r in rows:
            line = _clean_json_line(r)
            if fh is None or (used > 0 and used + len(line) > shard_bytes):
                if fh is not None:
                    fh.close()
                fh = (split_dir / f"{lang}-{shard_idx:03d}.jsonl").open("w", encoding="utf-8")
                shard_idx += 1
                used = 0
            fh.write(line)
            used += len(line)
        if fh is not None:
            fh.close()
        total += len(rows)
        print(f"[build] {split}/{lang}: {len(rows)} rows ({shard_idx} shard(s))")
    print(f"[build] {split}: {len(records)} problems -> {total} rows total")


def write_readme(out, reports):
    splits = "\n".join(
        f"- `{split}`: {sum(not e['drop'] for e in report)} problems x {len(LANGUAGES)} languages = "
        f"{sum(not e['drop'] for e in report) * len(LANGUAGES)} rows ({len(report)} in the source dataset)"
        for split, report in reports.items())
    (out / "README.md").write_text("""---
license: apache-2.0
language:
- code
task_categories:
- text-generation
---

# LeetCode multilingual benchmark dataset

LeetCode problems for the msl-multilingual-self-learning benchmark, flattened to
one row per (problem, language) in 9 languages, stored as `data/<split>/<lang>-NNN.jsonl`.
Each row has the `interface` for its `language` (from LeetCode's code snippets),
the shared `canonical_tests` (Python asserts from newfacade/LeetCodeDataset),
the `problem_description` (from the LeetCode page) and `metadata`.

Tests that break the problem's Constraints, do not fit a declared type in some
language, or expect inf/nan were removed; problems that take trees or linked
lists, modify their input in place, or kept fewer than 10 tests were dropped.
The test split also drops problems with several valid answers or whose text
refers to a figure. Where answers may come in any order or are decimals,
the asserts call `answers_match` (defined at the top of the test source), and
`canonical_tests.comparison` says which rule applies ("unordered", "float" or "exact").
`public_tests` holds the problem page's examples in the same form as
`canonical_tests` (the ones among the cleaned tests), and `hints` the page's
hints as plain text.
`reports/dropped.md` lists every dropped problem and test count, and
`reports/<split>.json` has the details.

## Splits

{splits}

## Load

```python
from datasets import load_dataset
train = load_dataset("neulab/leetcode", split="train")
test  = load_dataset("neulab/leetcode", split="test")
python_rows = test.filter(lambda r: r["language"] == "python")
```
""".replace("{splits}", splits))


def main():
    ap = argparse.ArgumentParser(
        description="Prep LeetCode splits into per-language JSONL for the HF "
                    "dataset push (throttled LeetCode fetch; resumable via cache)."
    )
    ap.add_argument("--out", type=Path, default=Path("/home/aci18914wh/leetcode"))
    ap.add_argument("--only", choices=SPLITS, default=None,
                    help="Build only this split (default: all splits).")
    ap.add_argument("--rate-delay", type=float, default=3.0,
                    help="Seconds between LeetCode requests (default 3.0).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N problems (smoke-test a subset).")
    ap.add_argument("--shard-mb", type=float, default=8.0,
                    help="Target max MiB per output file (default 8, under HF's ~10 "
                         "MiB ceiling); rows are sliced on boundaries.")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                    help="Where fetched LeetCode pages are cached.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Write only reports/ (what would be dropped), not the data files.")
    args = ap.parse_args()
    splits = (args.only,) if args.only else SPLITS

    records, reports = {}, {}
    for split in splits:
        records[split], reports[split] = build_problems_hf(
            split, cache_dir=args.cache_dir, rate_delay=args.rate_delay, limit=args.limit)
    if "train" in reports and "test" in reports:
        mark_overlap(reports["train"], reports["test"])
    for split in splits:
        records[split] = apply_drop_categories(split, records[split], reports[split])

    report_dir = args.out / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        (report_dir / f"{split}.json").write_text(json.dumps(reports[split], indent=1) + "\n")
        if not args.dry_run:
            build(split, records[split], args.out, shard_mb=args.shard_mb)
    if not args.dry_run:
        write_readme(args.out, reports)
    (report_dir / "dropped.md").write_text(drop_list(reports))
    summary = "\n\n".join(summarize(split, reports[split], len(LANGUAGES)) for split in splits)
    (report_dir / "summary.txt").write_text(summary + "\n")
    print(summary)


# --- raw LeetCode row loading (prep) ----------------------------------------


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
