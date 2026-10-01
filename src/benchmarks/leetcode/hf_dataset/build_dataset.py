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
* problems left with fewer than MIN_VALID_TESTS tests are dropped;
* per split, DROP_CATEGORIES: the test split drops problems with several valid
  answers or whose text refers to a figure; train drops problems LeetCode
  links to a test problem.

Answers that may come in any order, or are decimals, keep their tests but the
asserts are rewritten to `answers_match(candidate(...), expected)` (defined in
the same source: top-level order ignored, or 1e-5 tolerance), and
canonical_tests.comparison records "unordered" / "float" / "exact".

Every drop is recorded with its reason in reports/<split>.json (listed in
reports/dropped.md), and problems
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
import ast
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

# Hand-reviewed answer kinds where the wording patterns above get it wrong (read from the statement).
REVIEWED_ANSWERS = {
    # "any order" describes the input or the process; the answer's order matters.
    950: ("ordered", "the deck order is the answer"),
    2094: ("ordered", "'any arbitrary order' describes the digits; the answer must be sorted"),
    2197: ("ordered", "'in any arbitrary order' describes the replacements; the final array's order matters"),
    2273: ("ordered", "'in any arbitrary order' describes the operations; the final list's order matters"),
    # Both the groups and the items inside each group may come in any order.
    49: ("unordered_groups", "groups and the strings inside each group may come in any order"),
    609: ("unordered_groups", "groups and the paths inside each group may come in any order"),
    # Several different answers are correct.
    1030: ("multiple_answers", "cells sorted by distance; ties may come in any order"),
    1743: ("multiple_answers", "the array or its reverse; 'any order' describes the input pairs"),
    2215: ("multiple_answers", "two lists in fixed order, each list's items in any order"),
    1489: ("multiple_answers", "[critical, pseudo-critical], each list's indices in any order"),
    1125: ("multiple_answers", "any sufficient team of the smallest size"),
    2178: ("multiple_answers", "any split into the most unique even integers"),
    2699: ("multiple_answers", "any valid assignment of the modified edge weights"),
}


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


def _flags(qid: int, text: str, interfaces: dict[str, Any], meta: dict[str, Any], images: int) -> list[str]:
    flags = [name for name, pattern in FLAG_PATTERNS.items() if pattern.search(text)]
    # "rearrange the substrings in any order" describes the task, not the answer, unless a list is returned.
    if "any_order" in flags and not interfaces.get("python", {}).get("return_type", "").lower().startswith("list"):
        flags.remove("any_order")
    returns = [meta.get("return", {}).get("type", "")]
    returns += [i["return_type"] for i in interfaces.values()]
    if any(DECIMAL_TYPES.search(t) for t in returns):
        flags.append("decimal_answer")
    review = REVIEWED_ANSWERS.get(qid, (None, ""))[0]
    if review == "ordered" and "any_order" in flags:
        flags.remove("any_order")
        flags.append("any_order_wording_not_about_answer")
    elif review == "unordered_groups":
        flags = [f for f in flags if f != "any_order"] + ["unordered_groups"]
    elif review == "multiple_answers":
        flags = [f for f in flags if f != "any_order"]
        if "multiple_answers" not in flags:
            flags.append("multiple_answers")
    if images:
        flags.append("has_images")
    if meta.get("manual"):
        flags.append("leetcode_manual_judge")
    return flags


# --- Answer comparison --------------------------------------------------------------
# Problems whose answers an exact `==` would wrongly reject get their asserts rewritten to
# `assert answers_match(candidate(...), expected)`, with the helper defined in the same
# source, so anyone running check(candidate) compares correctly.

COMPARISON_HELPERS = {
    "unordered": '''def answers_match(result, expected):
    """The answer may list its items in any order (only the top-level order is free)."""
    if isinstance(result, list) and isinstance(expected, list):
        return sorted(result, key=repr) == sorted(expected, key=repr)
    return result == expected
''',
    "unordered_groups": '''def answers_match(result, expected):
    """The groups, and the items inside each group, may come in any order."""
    def canonical(groups):
        return sorted((sorted(g, key=repr) if isinstance(g, list) else g for g in groups), key=repr)
    if isinstance(result, list) and isinstance(expected, list):
        return canonical(result) == canonical(expected)
    return result == expected
''',
    "float": '''def answers_match(result, expected):
    """Decimal answers are accepted within 1e-5 (absolute or relative), as on LeetCode."""
    import math
    if isinstance(result, list) and isinstance(expected, list):
        return len(result) == len(expected) and all(answers_match(r, e) for r, e in zip(result, expected))
    numbers = (int, float)
    if isinstance(result, numbers) and isinstance(expected, numbers) and not isinstance(result, bool):
        return math.isclose(result, expected, rel_tol=1e-5, abs_tol=1e-5)
    return result == expected
''',
}


def comparison_for(flags: list[str]) -> str:
    if "decimal_answer" in flags:
        return "float"
    if "unordered_groups" in flags:
        return "unordered_groups"
    if "any_order" in flags:
        return "unordered"
    return "exact"


class _UseAnswersMatch(ast.NodeTransformer):
    def visit_Assert(self, node: ast.Assert) -> ast.Assert:
        test = node.test
        if (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
                and isinstance(test.left, ast.Call) and getattr(test.left.func, "id", "") == "candidate"):
            node.test = ast.Call(ast.Name("answers_match", ast.Load()), [test.left, test.comparators[0]], [])
        return node


def apply_comparison(source: str, comparison: str) -> str:
    """The test source with its asserts using the comparison's answers_match (unchanged for exact)."""
    if comparison == "exact":
        return source
    tree = _UseAnswersMatch().visit(ast.parse(source))
    return COMPARISON_HELPERS[comparison] + "\n\n" + ast.unparse(tree) + "\n"


# --- Build ----------------------------------------------------------------------

def build_problem(row: dict[str, Any], cache_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """(record or None if dropped, report entry) for one newfacade row."""
    qid = int(row["question_id"])
    crawled = load_cached(qid, cache_dir)
    question = crawled["question"] if crawled else None
    meta = _meta(question or {})
    entry: dict[str, Any] = {"question_id": qid, "task_id": row["task_id"], "drop": _problem_drops(question, meta)}
    if question is not None:
        try:
            entry["similar"] = [q["titleSlug"] for q in json.loads(question.get("similarQuestions") or "[]")]
        except ValueError:
            entry["similar"] = []
    if entry["drop"] in (["not_crawled"], ["premium"]):
        return None, entry

    description = html_to_text(question["content"])
    interfaces, interface_problems = _interfaces(qid, question)
    if interface_problems:
        entry["drop"].append("interface")
        entry["interface_problems"] = interface_problems
    names = [p.name for p in parse_hf_python_signature(row["starter_code"]).parameters]
    entry["flags"] = _flags(qid, description.text, interfaces, meta, len(description.images) + description.videos)

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
    entry["comparison"] = comparison_for(entry["flags"])
    if entry["drop"]:
        return None, entry

    record = {
        "question_id": qid,
        "task_id": row["task_id"],
        "difficulty": question.get("difficulty") or row["difficulty"],
        "problem_description": description.text,
        "interfaces": interfaces,
        "canonical_tests": {
            "language": "python",
            "source": apply_comparison(source, entry["comparison"]),
            "comparison": entry["comparison"],
        },
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


CATEGORY_NOTES = {
    "any_order": "list answer in any order -> 'unordered' comparison",
    "unordered_groups": "groups and items in any order -> 'unordered_groups' comparison",
    "any_order_wording_not_about_answer": "'any order' wording, answer order matters -> exact",
    "multiple_answers": "several valid answers; exact comparison rejects some",
    "decimal_answer": "float/double answer -> 'float' comparison (1e-5)",
    "decimal_tolerance": "statement states a decimal tolerance",
    "has_images": "page has images/videos (dropped from the text)",
    "figure_reference": "text refers to a figure",
    "leetcode_manual_judge": "LeetCode marks the judge as manual",
    "unchecked_constraints": "has constraint rules the audit cannot check",
    "similar_to_test": "LeetCode lists a test-split problem as similar",
}


def _ids(entries: list[dict], limit: int = 25) -> str:
    ids = sorted(e["question_id"] for e in entries)
    return " ".join(map(str, ids[:limit])) + (" ..." if len(ids) > limit else "")


def summarize(split: str, report: list[dict]) -> str:
    """The counts to quote: problems, tests and (problem, language) rows dropped or flagged, by category."""
    n_lang = len(LANGUAGES)
    kept = [e for e in report if not e["drop"]]
    lines = [f"== {split}: {len(report)} source problems -> {len(kept)} kept "
             f"({len(kept) * n_lang} rows = problems x {n_lang} languages)", "",
             "  Problems dropped (first reason = counted once / any reason = overlapping):"]
    first = Counter(e["drop"][0] for e in report if e["drop"])
    anywhere = Counter(r for e in report for r in e["drop"])
    for reason in ("not_crawled", "premium", "class_design", "tree_or_linked_list", "in_place", "interface",
                   "too_few_tests", *dict.fromkeys(c for cs in DROP_CATEGORIES.values() for c in cs)):
        if anywhere[reason]:
            lines.append(f"    {reason:30} {first[reason]:5} / {anywhere[reason]:5}")
    few = [e for e in report if e["drop"] == ["too_few_tests"]]
    lines.append(f"    too_few_tests only: {len(few)} problems "
                 f"({sum(1 for e in few if e['tests']['total'] < MIN_VALID_TESTS)} had < {MIN_VALID_TESTS} "
                 f"tests to begin with, {sum(1 for e in few if e['tests']['total'] >= MIN_VALID_TESTS)} "
                 f"fell below after test cleaning)")

    tested = [e for e in report if "tests" in e]
    lines += ["", f"  Tests (all {len(tested)} problems whose tests were checked / the {len(kept)} kept problems):"]
    for group in ("total", "constraint_violation", "int_overflow", "non_finite"):
        def count(entries, group=group):
            if group == "total":
                return sum(e["tests"]["total"] for e in entries)
            return sum(e["tests"]["dropped"].get(group, 0) for e in entries)
        affected = sum(1 for e in kept if e["tests"]["dropped"].get(group))
        extra = f"   (in {affected} kept problems)" if group != "total" else ""
        lines.append(f"    {group:30} {count(tested):7} / {count(kept):7}{extra}")

    lines += ["", "  Categories among kept problems (not dropped; decide per category):",
              f"    {'category':36} {'problems':>8} {'rows':>6}  ids"]
    by_flag: dict[str, list[dict]] = {}
    for e in kept:
        flags = list(e.get("flags", []))
        if e.get("unchecked_constraints"):
            flags.append("unchecked_constraints")
        if e.get("similar_to_test"):
            flags.append("similar_to_test")
        for f in flags:
            by_flag.setdefault(f, []).append(e)
    for flag in CATEGORY_NOTES:
        entries = by_flag.get(flag, [])
        if entries:
            lines.append(f"    {flag:36} {len(entries):8} {len(entries) * n_lang:6}  {_ids(entries)}")
    lines += ["", "  Category meanings:"] + [f"    {k}: {v}" for k, v in CATEGORY_NOTES.items()]
    return "\n".join(lines)


def mark_overlap(train: list[dict], test: list[dict]) -> None:
    """Record the test problems LeetCode links to each train problem (listed on either page)."""
    for e in train:
        hits = [f"{t['question_id']} {t['task_id']}" for t in test
                if t["task_id"] in e.get("similar", []) or e["task_id"] in t.get("similar", [])
                or t["question_id"] == e["question_id"]]
        if hits:
            e["similar_to_test"] = hits


# Categories dropped per split, decided on the dry-run report (2026-09-30). The test split
# keeps only problems we can grade correctly in every language; train keeps what can still
# give a learning signal but must not overlap the test split.
DROP_CATEGORIES = {
    "test": (
        "multiple_answers",   # no general way to accept every valid answer
        "figure_reference",   # the text points to a figure the model cannot see
    ),
    "train": (
        "similar_to_test",    # LeetCode links it to a test problem: keep train and test apart
    ),
}


def apply_drop_categories(split: str, records: list[dict], report: list[dict]) -> list[dict]:
    """Drop the kept problems in this split's DROP_CATEGORIES; returns the remaining records."""
    dropped = set()
    for e in report:
        if e["drop"]:
            continue
        categories = set(e.get("flags", [])) | ({"similar_to_test"} if e.get("similar_to_test") else set())
        reasons = [c for c in DROP_CATEGORIES.get(split, ()) if c in categories]
        if reasons:
            e["drop"].extend(reasons)
            dropped.add(e["question_id"])
    return [r for r in records if r["question_id"] not in dropped]


def drop_list(reports: dict[str, list[dict]]) -> str:
    """Markdown list of every dropped problem, and of the tests dropped from kept problems."""
    lines = ["# Dropped from neulab/leetcode", ""]
    for split, report in reports.items():
        dropped = [e for e in report if e["drop"]]
        kept = [e for e in report if not e["drop"]]
        lines += [f"## {split}: {len(dropped)} of {len(report)} problems dropped, {len(kept)} kept", "",
                  "| question_id | task_id | reasons |", "|---|---|---|"]
        lines += [f"| {e['question_id']} | {e['task_id']} | {', '.join(e['drop'])} |"
                  for e in sorted(dropped, key=lambda e: e["question_id"])]
        cleaned = [e for e in kept if e["tests"]["dropped"]]
        total = sum(sum(e["tests"]["dropped"].values()) for e in cleaned)
        lines += ["", f"### {split}: tests dropped from kept problems ({total} tests in {len(cleaned)} problems)", "",
                  "| question_id | tests | kept | constraint_violation | int_overflow | non_finite |",
                  "|---|---|---|---|---|---|"]
        for e in sorted(cleaned, key=lambda e: e["question_id"]):
            d = e["tests"]["dropped"]
            lines.append(f"| {e['question_id']} | {e['tests']['total']} | {e['tests']['kept']} | "
                         f"{d.get('constraint_violation', 0)} | {d.get('int_overflow', 0)} | {d.get('non_finite', 0)} |")
        lines.append("")
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
The test split also drops problems with several valid answers or whose text
refers to a figure; train drops problems LeetCode links to a test problem. Where answers may come in any order or are decimals,
the asserts call `answers_match` (defined at the top of the test source), and
`canonical_tests.comparison` says which rule applies ("unordered", "float" or "exact").
`reports/dropped.md` lists every dropped problem and test count, and
`reports/<split>.json` has the details.

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

    reports: dict[str, list[dict]] = {}
    records: dict[str, list[dict]] = {}
    for split in (args.only,) if args.only else SPLITS:
        records[split], reports[split] = build_split(split, args.cache_dir, args.limit)
    if "train" in reports and "test" in reports:
        mark_overlap(reports["train"], reports["test"])
    for split in reports:
        records[split] = apply_drop_categories(split, records[split], reports[split])
    report_dir = args.out / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for split, report in reports.items():
        (report_dir / f"{split}.json").write_text(json.dumps(report, indent=1) + "\n")
        summaries.append(summarize(split, report))
        if not args.dry_run:
            write_split(split, records[split], args.out, args.shard_mb)
    if not args.dry_run:
        write_readme(args.out)
    (report_dir / "dropped.md").write_text(drop_list(reports))
    summary = "\n\n".join(summaries)
    (report_dir / "summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
