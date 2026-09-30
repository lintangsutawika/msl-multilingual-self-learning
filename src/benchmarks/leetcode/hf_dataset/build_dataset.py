"""Build the neulab/leetcode Hugging Face dataset.

This is the PREP step (build once, push to HF). For every problem in
newfacade/LeetCodeDataset it takes the canonical tests (inputs + the Python
reference's outputs) and combines them with the LeetCode page crawled by
crawl.py: the description (with exponents, subscripts and lists kept, see
description.py), the per-language interfaces (from LeetCode's codeSnippets)
and LeetCode's typed metaData. Then it cleans:

* problems are dropped when they are premium (no statement), take or return a
  TreeNode/ListNode (no transport for all 9 languages yet), modify their input
  in place (newfacade's tests only check `== None`), or lack a parseable
  interface in one of the 9 languages;
* tests are dropped when they break the problem's current Constraints, do not
  fit a declared type in some language, or expect inf/nan (constraints.py);
* problems left with fewer than MIN_VALID_TESTS tests are dropped.

Every drop is recorded with its reason in reports/<split>.json, and problems
whose answers need more than an exact comparison (any order, several valid
answers, decimals) are flagged there. The output is one row per (problem,
language) in data/<split>/<lang>-NNN.jsonl (row-boundary shards so every file
stays under HF's ~10 MiB per-file ceiling; HF globs them into one split).

    uv run python -m src.benchmarks.leetcode.hf_dataset.crawl
    uv run python -m src.benchmarks.leetcode.hf_dataset.build_dataset --out ../leetcode-hf --dry-run
    uv run python -m src.benchmarks.leetcode.hf_dataset.build_dataset --out ../leetcode-hf

After `git push` of --out, `datasets.load_dataset("neulab/leetcode", split=...)`
returns the rows, which dataset.py pulls at generation time.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .constraints import MIN_VALID_TESTS, audit_problem, filter_canonical_tests
from .crawl import DEFAULT_CACHE_DIR, SOURCE_DATASET, load_cached
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
SPLITS = ("train", "test")

# LeetCode metaData types that JSON test values cannot carry to every language yet.
UNSUPPORTED_TYPES = re.compile(r"\b(TreeNode|ListNode|Node)\b")

# Statements whose answer an exact comparison may wrongly reject.
FLAG_PATTERNS = {
    "any_order": re.compile(r"\bin any (?:arbitrary )?order\b", re.I),
    "multiple_answers": re.compile(
        r"\b(?:return|output|print) any (?:of them|one of them|valid|such|possible)\b"
        r"|\bif there are (?:multiple|several|many) (?:valid |possible )?(?:answers|solutions|results)\b"
        r"|\bany (?:valid )?(?:answer|solution) (?:will be|is) accepted\b"
        r"|\b(?:multiple|several) (?:valid|possible) (?:answers|solutions)\b",
        re.I,
    ),
    "decimal_tolerance": re.compile(r"\bwithin 10\^-?\d+ of the actual\b|\bare accepted\b.*\b10\^-\d+", re.I),
    "figure_reference": re.compile(
        r"\b(?:as |is )?(?:shown|illustrated|depicted|pictured) (?:in|below|above)\b"
        r"|\bin the (?:figure|image|picture|diagram|illustration)\b"
        r"|\b(?:figure|image|picture|diagram) (?:below|above)\b",
        re.I,
    ),
}
DECIMAL_TYPES = re.compile(r"\b(float|double)\b")


# --- Problem-level checks -----------------------------------------------------

def _meta(question: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(question.get("metaData") or "{}")
    except ValueError:
        return {}


def _interfaces(question_id: int, question: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """({language: interface dict}, [problems]) from the crawled codeSnippets."""
    snippets = {s["langSlug"]: s["code"] for s in question.get("codeSnippets") or []}
    interfaces, problems = {}, []
    for language in LANGUAGES:
        code = snippets.get(LANGUAGE_SLUGS[language])
        if not code:
            problems.append(f"{language}: no code snippet")
            continue
        try:
            interface = parse_leetcode_interface(question_id, language, code)
        except Exception as exc:  # an unusual signature form: skip the problem, report it
            problems.append(f"{language}: {type(exc).__name__}: {exc}"[:200])
            continue
        if interface is None:
            problems.append(f"{language}: unparsed signature")
            continue
        interfaces[language] = {
            "raw_signature": interface.raw_signature,
            "container": interface.container,
            "callable": interface.callable_name,
            "parameters": [{"name": p.name, "type": p.type} for p in interface.parameters],
            "return_type": interface.return_type,
            "signature_source": "leetcode/codeSnippets",
            "type_source": "leetcode/codeSnippets",
        }
    return interfaces, problems


def _problem_drops(question: dict[str, Any] | None, meta: dict[str, Any]) -> list[str]:
    """Reasons the whole problem cannot be used, whatever its tests."""
    if question is None:
        return ["not_crawled"]
    if question.get("isPaidOnly") or not question.get("content"):
        return ["premium"]
    reasons = []
    if "classname" in meta or meta.get("systemdesign"):
        reasons.append("class_design")
    types = [p.get("type", "") for p in meta.get("params", [])] + [meta.get("return", {}).get("type", "")]
    if any(UNSUPPORTED_TYPES.search(t) for t in types):
        reasons.append("tree_or_linked_list")
    if meta.get("return", {}).get("type") == "void" or "output" in meta:
        reasons.append("in_place")
    return reasons


def _flags(text: str, interfaces: dict[str, Any], meta: dict[str, Any], images: int) -> list[str]:
    flags = [name for name, pattern in FLAG_PATTERNS.items() if pattern.search(text)]
    returns = [meta.get("return", {}).get("type", "")]
    returns += [i["return_type"] for i in interfaces.values()]
    if any(DECIMAL_TYPES.search(t) for t in returns):
        flags.append("decimal_answer")
    if images:
        flags.append("has_images")
    if meta.get("manual"):
        flags.append("leetcode_manual_judge")
    return flags


# --- Build ----------------------------------------------------------------------

def build_problem(row: dict[str, Any], cache_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """(record or None if dropped, report entry) for one newfacade row."""
    qid = int(row["question_id"])
    crawled = load_cached(qid, cache_dir)
    question = crawled["question"] if crawled else None
    meta = _meta(question or {})
    entry: dict[str, Any] = {"question_id": qid, "task_id": row["task_id"], "drop": _problem_drops(question, meta)}
    if entry["drop"] in (["not_crawled"], ["premium"]):
        return None, entry

    description = html_to_text(question["content"])
    interfaces, interface_problems = _interfaces(qid, question)
    if interface_problems:
        entry["drop"].append("interface")
        entry["interface_problems"] = interface_problems
    names = [p.name for p in parse_hf_python_signature(row["starter_code"]).parameters]
    entry["flags"] = _flags(description.text, interfaces, meta, len(description.images) + description.videos)

    # Tests: audit against the current Constraints, then drop the unfair ones.
    problem = {
        "question_id": qid,
        "metadata": {"canonical_parameter_names": names},
        "canonical_tests": {"language": "python", "source": row["test"]},
        "interfaces": interfaces,
    }
    audit = audit_problem(problem, question["content"])
    invalid = {t["test"]: t["violates"] for t in audit["invalid_tests"]}
    source, dropped, kept = filter_canonical_tests(problem, invalid)
    reasons = Counter()
    for d in dropped:
        if "violates" in d:
            reasons["constraint_violation"] += 1
        elif set(d["unrepresentable_in"]) >= set(interfaces):
            reasons["non_finite"] += 1
        else:
            reasons["int_overflow"] += 1
    entry["tests"] = {"total": kept + len(dropped), "kept": kept, "dropped": dict(reasons)}
    entry["dropped_tests"] = dropped
    entry["unchecked_constraints"] = audit["unchecked"]
    if kept < MIN_VALID_TESTS:
        entry["drop"].append("too_few_tests")
    if entry["drop"]:
        return None, entry

    record = {
        "question_id": qid,
        "task_id": row["task_id"],
        "difficulty": question.get("difficulty") or row["difficulty"],
        "problem_description": description.text,
        "interfaces": interfaces,
        "canonical_tests": {"language": "python", "source": source},
        "metadata": {
            "canonical_parameter_names": names,
            "leetcode_dataset_entry_point": row["entry_point"],
            "problem_source": SOURCE_DATASET,
            "description_source": "leetcode.com",
            "interface_source": "leetcode/codeSnippets",
            "fetched_at": crawled["fetched_at"],
            "tests_total": entry["tests"]["total"],
            "tests_dropped": len(dropped),
            "flags": entry["flags"],
            "topic_tags": [t["slug"] for t in question.get("topicTags") or []],
        },
    }
    return record, entry


def build_split(split: str, cache_dir: Path, limit: int | None = None) -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset

    rows = list(load_dataset(SOURCE_DATASET, split=split))
    if limit is not None:
        rows = rows[:limit]
    records, report = [], []
    for row in rows:
        record, entry = build_problem(row, cache_dir)
        report.append(entry)
        if record is not None:
            records.append(record)
    return records, report


def summarize(split: str, report: list[dict]) -> str:
    """The counts to quote: problems, tests and (problem, language) rows dropped, by reason."""
    n = len(report)
    kept = [e for e in report if not e["drop"]]
    lines = [f"== {split}: {n} source problems -> {len(kept)} kept "
             f"({len(kept) * len(LANGUAGES)} rows = problems x {len(LANGUAGES)} languages)"]
    first = Counter(e["drop"][0] for e in report if e["drop"])
    anywhere = Counter(r for e in report for r in e["drop"])
    lines.append("  problems dropped, by reason (first reason / any reason):")
    for reason in ("not_crawled", "premium", "class_design", "tree_or_linked_list", "in_place", "interface",
                   "too_few_tests"):
        if anywhere[reason]:
            lines.append(f"    {reason:30} {first[reason]:5} / {anywhere[reason]}")
    tested = [e for e in report if "tests" in e]
    total = sum(e["tests"]["total"] for e in tested)
    dropped = Counter()
    for e in tested:
        dropped.update(e["tests"]["dropped"])
    lines.append(f"  tests (problems whose tests were checked: {len(tested)}): {total} total, "
                 f"{sum(dropped.values())} dropped")
    for reason, count in dropped.most_common():
        lines.append(f"    {reason:30} {count:5}")
    kept_tests = sum(e["tests"]["kept"] for e in kept)
    kept_dropped = sum(sum(e["tests"]["dropped"].values()) for e in kept)
    lines.append(f"  in kept problems: {kept_tests} tests kept, {kept_dropped} dropped")
    flags = Counter(f for e in kept for f in e.get("flags", []))
    lines.append("  kept problems flagged for review:")
    for flag, count in flags.most_common():
        lines.append(f"    {flag:30} {count:5}")
    unchecked = sum(len(e.get("unchecked_constraints", [])) for e in kept)
    lines.append(f"  constraint rules not checked automatically (kept problems): {unchecked}")
    return "\n".join(lines)


# --- Output ---------------------------------------------------------------------

def flatten(records: list[dict]) -> list[dict]:
    """Expand nested problem-records into one row per (problem, language)."""
    flat = []
    for rec in records:
        base = {k: v for k, v in rec.items() if k != "interfaces"}
        for language, interface in rec["interfaces"].items():
            flat.append({**base, "language": language, "interface": interface})
    return flat


def _clean_json_line(r: dict) -> str:
    """Serialize a row to a single JSONL line, escaping raw line-separator chars
    (U+0085/U+2028/U+2029/CR) that HF's line reader would otherwise split on."""
    line = json.dumps(r, ensure_ascii=False)
    for ch, esc in (("\x85", "\\u0085"), (" ", "\\u2028"), (" ", "\\u2029"), ("\r", "\\r")):
        line = line.replace(ch, esc)
    return line + "\n"


def write_split(split: str, records: list[dict], out: Path, shard_mb: float = 8) -> None:
    split_dir = out / "data" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for old in split_dir.glob("*.jsonl"):
        old.unlink()
    by_lang: dict[str, list[dict]] = {}
    for row in flatten(records):
        by_lang.setdefault(row["language"], []).append(row)
    shard_bytes = shard_mb * 1024 * 1024
    for lang, rows in sorted(by_lang.items()):
        shard_idx, used, fh = 0, 0, None
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
        print(f"[build] {split}/{lang}: {len(rows)} rows in {shard_idx} shard(s)")


def write_readme(out: Path) -> None:
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
`reports/<split>.json` lists every dropped problem and test with its reason.

## Load

```python
from datasets import load_dataset
train = load_dataset("neulab/leetcode", split="train")
test  = load_dataset("neulab/leetcode", split="test")
```
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="Local clone of the HF dataset repo.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="crawl.py's cache.")
    parser.add_argument("--only", choices=SPLITS, default=None, help="Build only this split.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N problems.")
    parser.add_argument("--shard-mb", type=float, default=8.0, help="Max MiB per data file.")
    parser.add_argument("--dry-run", action="store_true", help="Write only the reports, not the data files.")
    args = parser.parse_args()

    summaries = []
    for split in (args.only,) if args.only else SPLITS:
        records, report = build_split(split, args.cache_dir, args.limit)
        reports = args.out / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        (reports / f"{split}.json").write_text(json.dumps(report, indent=1) + "\n")
        summaries.append(summarize(split, report))
        if not args.dry_run:
            write_split(split, records, args.out, args.shard_mb)
    if not args.dry_run:
        write_readme(args.out)
    summary = "\n\n".join(summaries)
    (args.out / "reports" / "summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
