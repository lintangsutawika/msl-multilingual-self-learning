"""Decide what the build keeps: which problems, which tests, and how answers are compared.

* Problems are dropped when they are premium (no statement), take or return a
  TreeNode/ListNode (no transport for all 9 languages yet), modify their input
  in place (newfacade's tests only check `== None`), or are class designs.
* Tests are dropped when they break the problem's current Constraints, do not
  fit a declared type in some language, or expect inf/nan (constraints.py);
  problems left with fewer than MIN_VALID_TESTS tests are dropped.
* Answers that may come in any order, or are decimals, keep their tests, but
  the asserts are rewritten to `answers_match(candidate(...), expected)`,
  defined in the same test source, so anyone running check(candidate)
  compares correctly.
* DROP_CATEGORIES drops further categories per split.

Every decision is recorded in a report entry per problem; summarize() and
drop_list() turn the entries into reports/summary.txt and reports/dropped.md.
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter
from typing import Any

from .constraints import MIN_VALID_TESTS, audit_problem, filter_canonical_tests

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
    "figure_reference": re.compile(
        r"\b(?:as |is )?(?:shown|illustrated|depicted|pictured) (?:in|below|above)\b"
        r"|\bin the (?:figure|image|picture|diagram|illustration)\b"
        r"|\b(?:figure|image|picture|diagram) (?:below|above)\b",
        re.I,
    ),
}
DECIMAL_TYPES = re.compile(r"\b(float|double)\b")

# Read from the statements: "any order" wording where the answer's (top-level) order
# still matters, so an order-free comparison would accept wrong answers.
ORDER_MATTERS = {
    950,   # the deck order is the answer
    2094,  # "any arbitrary order" describes the digits; the answer must be sorted
    2197,  # "any arbitrary order" describes the replacements; the final array's order matters
    2273,  # "any arbitrary order" describes the operations; the final list's order matters
    1030,  # cells sorted by distance (only ties are free)
    1743,  # "any order" describes the input pairs; the answer is an ordered array
    2215,  # [answer0, answer1] in fixed order; only each list's items are free
    1489,  # [critical, pseudo-critical] in fixed order; only each list's items are free
}
# Both the groups and the items inside each group may come in any order.
UNORDERED_GROUPS = {49, 609}

# Categories dropped per split, decided on the dry-run report (2026-09-30). The test split
# keeps only problems we can grade correctly in every language; train keeps everything that
# can still give a learning signal (including problems LeetCode lists as "similar" to a test
# problem: the links mean related topic, not the same problem).
DROP_CATEGORIES = {
    "test": (
        "multiple_answers",   # no general way to accept every valid answer
        "figure_reference",   # the text points to a figure the model cannot see
    ),
    "train": (),
}


# --- Problems -----------------------------------------------------------------

def problem_drops(question: dict[str, Any] | None) -> list[str]:
    """Reasons the whole problem cannot be used, whatever its tests."""
    if question is None:
        return ["not_on_leetcode"]
    if question.get("isPaidOnly") or not question.get("content"):
        return ["premium"]
    meta = metadata(question)
    reasons = []
    if "classname" in meta or meta.get("systemdesign"):
        reasons.append("class_design")
    types = [p.get("type", "") for p in meta.get("params", [])] + [meta.get("return", {}).get("type", "")]
    if any(UNSUPPORTED_TYPES.search(t) for t in types):
        reasons.append("tree_or_linked_list")
    if meta.get("return", {}).get("type") == "void" or "output" in meta:
        reasons.append("in_place")
    return reasons


def metadata(question: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(question.get("metaData") or "{}")
    except ValueError:
        return {}


def flags(question_id: int, text: str, interfaces: dict[str, Any], question: dict[str, Any],
          media: int) -> list[str]:
    """What a reviewer should know about the problem's answer and statement."""
    found = [name for name, pattern in FLAG_PATTERNS.items() if pattern.search(text)]
    # "rearrange the substrings in any order" describes the task, not the answer, unless a list is returned.
    returns_list = interfaces.get("python", {}).get("return_type", "").lower().startswith("list")
    if "any_order" in found and (not returns_list or question_id in ORDER_MATTERS):
        found.remove("any_order")
    if question_id in UNORDERED_GROUPS:
        found = [f for f in found if f != "any_order"] + ["unordered_groups"]
    returns = [metadata(question).get("return", {}).get("type", "")]
    returns += [i["return_type"] for i in interfaces.values()]
    if any(DECIMAL_TYPES.search(t) for t in returns):
        found.append("decimal_answer")
    if media:
        found.append("has_images")
    return found


# --- Tests ----------------------------------------------------------------------

def clean_tests(problem: dict[str, Any], content: str) -> tuple[str, dict[str, Any]]:
    """(test source without unfair tests, report fields) for a problem's newfacade tests."""
    audit = audit_problem(problem, content)
    invalid = {t["test"]: t["violates"] for t in audit["invalid_tests"]}
    source, dropped, kept = filter_canonical_tests(problem, invalid)
    reasons = Counter()
    for d in dropped:
        if "violates" in d:
            reasons["constraint_violation"] += 1
        elif set(d["unrepresentable_in"]) >= set(problem["interfaces"]):
            reasons["non_finite"] += 1
        else:
            reasons["int_overflow"] += 1
    return source, {
        "tests": {"total": kept + len(dropped), "kept": kept, "dropped": dict(reasons)},
        "dropped_tests": dropped,
        "unchecked_constraints": audit["unchecked"],
        "drop": ["too_few_tests"] if kept < MIN_VALID_TESTS else [],
    }


# --- Answer comparison ------------------------------------------------------------

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


def comparison_for(found: list[str]) -> str:
    if "decimal_answer" in found:
        return "float"
    if "unordered_groups" in found:
        return "unordered_groups"
    if "any_order" in found:
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


# --- Across splits --------------------------------------------------------------

def mark_overlap(train: list[dict], test: list[dict]) -> None:
    """Record the test problems LeetCode links to each train problem (listed on either page)."""
    for e in train:
        hits = [f"{t['question_id']} {t['task_id']}" for t in test
                if t["task_id"] in e.get("similar", []) or e["task_id"] in t.get("similar", [])
                or t["question_id"] == e["question_id"]]
        if hits:
            e["similar_to_test"] = hits


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


# --- Reports ----------------------------------------------------------------------

CATEGORY_NOTES = {
    "any_order": "list answer in any order -> 'unordered' comparison",
    "unordered_groups": "groups and items in any order -> 'unordered_groups' comparison",
    "multiple_answers": "several valid answers; exact comparison rejects some",
    "decimal_answer": "float/double answer -> 'float' comparison (1e-5)",
    "has_images": "page has images/videos (dropped from the text)",
    "figure_reference": "text refers to a figure",
    "unchecked_constraints": "has constraint rules the audit cannot check",
    "similar_to_test": "LeetCode lists a test-split problem as similar",
}
PROBLEM_DROPS = ("not_on_leetcode", "premium", "class_design", "tree_or_linked_list", "in_place",
                 "interface", "too_few_tests")


def _ids(entries: list[dict], limit: int = 25) -> str:
    ids = sorted(e["question_id"] for e in entries)
    return " ".join(map(str, ids[:limit])) + (" ..." if len(ids) > limit else "")


def summarize(split: str, report: list[dict], n_languages: int) -> str:
    """The counts to quote: problems, tests and (problem, language) rows dropped or flagged, by category."""
    kept = [e for e in report if not e["drop"]]
    lines = [f"== {split}: {len(report)} source problems -> {len(kept)} kept "
             f"({len(kept) * n_languages} rows = problems x {n_languages} languages)", "",
             "  Problems dropped (first reason = counted once / any reason = overlapping):"]
    first = Counter(e["drop"][0] for e in report if e["drop"])
    anywhere = Counter(r for e in report for r in e["drop"])
    for reason in (*PROBLEM_DROPS, *dict.fromkeys(c for cs in DROP_CATEGORIES.values() for c in cs)):
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

    lines += ["", "  Categories among kept problems:", f"    {'category':36} {'problems':>8} {'rows':>6}  ids"]
    by_flag: dict[str, list[dict]] = {}
    for e in kept:
        found = list(e.get("flags", []))
        if e.get("unchecked_constraints"):
            found.append("unchecked_constraints")
        if e.get("similar_to_test"):
            found.append("similar_to_test")
        for f in found:
            by_flag.setdefault(f, []).append(e)
    for flag in CATEGORY_NOTES:
        entries = by_flag.get(flag, [])
        if entries:
            lines.append(f"    {flag:36} {len(entries):8} {len(entries) * n_languages:6}  {_ids(entries)}")
    lines += ["", "  Category meanings:"] + [f"    {k}: {v}" for k, v in CATEGORY_NOTES.items()]
    return "\n".join(lines)


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
