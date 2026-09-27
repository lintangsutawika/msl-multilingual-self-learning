"""Audit canonical tests against LeetCode's own Constraints section.

LeetCodeDataset generated its test inputs and recorded the Python reference
solution's output, without checking the inputs against the problem's
constraints. A test that breaks the contract has no well-defined answer and
penalizes correct solutions that rely on the contract (fixed-size arrays in
typed languages, "an answer always exists", ...), so it is not a fair test
for any language.

Each <li> under "Constraints" on the LeetCode page becomes a check: common
forms (ranges, lengths, character sets, uniqueness, ...) are parsed
automatically; "The input is generated such that ..." guarantees have
hand-written checks in MANUAL_CHECKS; rules that cannot be checked without
solving the problem are listed in UNCHECKABLE. Rules matching none of these
are reported as unchecked so they can be added.

    uv run python -m src.benchmarks.leetcode.constraints --set b --fetch

writes benchmarks/leetcode/data/invalid_tests.json, which task generation
uses to drop invalid tests for every language.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import html
import itertools
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .adapter import _test_value, canonical_names

DEFAULT_DATASET = Path("benchmarks/leetcode/data/leetcode_multilingual_leetcode.jsonl")
DEFAULT_CONTENT_DIR = Path("benchmarks/leetcode/data/cache/leetcode_content")
DEFAULT_OUTPUT = Path("benchmarks/leetcode/data/invalid_tests.json")
SPLIT_DIR = Path("benchmarks/leetcode/data/splits")

Check = Callable[[dict[str, Any]], bool]
OPS = r"(<=|>=|==|!=|<|>)"
SAFE_BUILTINS = {
    "len": len, "all": all, "any": any, "range": range, "set": set, "sorted": sorted,
    "min": min, "max": max, "sum": sum, "abs": abs, "int": int, "floor": math.floor,
}


# --- Page text ---------------------------------------------------------------

def _plain(fragment: str) -> str:
    """HTML fragment -> plain text, with 10<sup>5</sup> -> 10**(5) and l<sub>i</sub> -> li."""
    fragment = re.sub(r"<sup>\s*(.*?)\s*</sup>", r"**(\1)", fragment, flags=re.S)
    fragment = re.sub(r"<[^>]+>", "", fragment)
    fragment = html.unescape(fragment).replace(" ", " ").replace("−", "-")
    return " ".join(fragment.split())


def constraint_items(content: str) -> list[str]:
    """Plain-text items of the page's Constraints list."""
    m = re.search(r"Constraints:(.*?)(</ul>|$)", content, re.S)
    if not m:
        return []
    return [_plain(item) for item in re.findall(r"<li>(.*?)</li>", m.group(1), re.S)]


def element_names(content: str, params: set[str]) -> dict[str, str]:
    """Names the statement gives to list elements: `queries[i] = [li, ri]` -> li = queries[i][0]."""
    names: dict[str, str] = {}
    for m in re.finditer(r"\b([A-Za-z_]\w*)\[i\]\s*==?\s*\[([^\]]+)\]", _plain(content)):
        if m.group(1) not in params:
            continue
        for pos, name in enumerate(n.strip() for n in m.group(2).split(",")):
            if re.fullmatch(r"[A-Za-z_]\w*", name) and name not in params:
                names.setdefault(name, f"{m.group(1)}[i][{pos}]")
    return names


# --- Automatic rules ---------------------------------------------------------

def _normalize(text: str, aliases: dict[str, str]) -> str:
    text = re.sub(r"(\b[A-Za-z_]\w*(?:\[[^\]]+\])*)\.length\b", r"len(\1)", text)
    for name, target in sorted(aliases.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"\b{re.escape(name)}\b", target, text)
    return text


def _index_vars(expr: str, params: set[str]) -> list[tuple[str, str]]:
    """Free index variables with the list each ranges over, outermost first."""
    found: dict[str, str] = {}
    for m in re.finditer(r"\[([A-Za-z_]\w*)\]", expr):
        var = m.group(1)
        if var in params or var in found:
            continue
        # The list being indexed is everything before this subscript, back to its name.
        head = re.search(r"([A-Za-z_]\w*(?:\[[^\[\]]+\])*)$", expr[:m.start()])
        if head:
            found[var] = head.group(1)
    return list(found.items())


def _compile(expr: str, params: set[str], extra: dict[str, Any] | None = None) -> Check:
    loops = _index_vars(expr, params)
    body = expr
    # x[i] != x[j] compares distinct positions only.
    for (v1, s1), (v2, s2) in itertools.combinations(loops, 2):
        if s1 == s2:
            body = f"({v1} == {v2}) or ({body})"
    for var, seq in reversed(loops):
        body = f"all(({body}) for {var} in range(len({seq})))"
    tree = ast.parse(body, mode="eval")
    allowed = params | {v for v, _ in loops} | set(SAFE_BUILTINS) | set(extra or {})
    unknown = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - allowed
    if unknown:
        raise ValueError(f"unknown names {sorted(unknown)}")
    code = compile(tree, "<constraint>", "eval")
    scope = {"__builtins__": {}, **SAFE_BUILTINS, **(extra or {})}
    return lambda env: eval(code, {**scope, **env})


def _split_top_level(text: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in text:
        depth += ch in "[("
        depth -= ch in "])"
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    out.append(cur.strip())
    return out


def _targets(text: str) -> list[str]:
    """`s and t`, `message[i] and bannedWords[i]`, `word1, word2` -> names."""
    return [t.strip() for t in re.split(r",|\band\b", text) if t.strip()]


def _is_prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))


def _learn_definitions(text: str, params: set[str], aliases: dict[str, str]) -> None:
    """Record names a rule defines: `1 <= n == nums.length`, `edges.length == n - 1`."""
    parts = re.split(OPS, text)
    if len(parts) < 3:
        return
    operands = [o.strip().rstrip(".") for o in parts[::2]]
    for a, op, b in zip(operands, parts[1::2], operands[1:]):
        if op != "==":
            continue
        for name, target in ((a, b), (b, a)):
            target_norm = _normalize(target, aliases)
            if re.search(r"\b[A-Za-z_]\w*\b", target_norm) is None:
                continue
            # A per-row length defines the row size from the first row.
            target_norm = re.sub(r"\[[a-z]\]", "[0]", target_norm)
            if re.fullmatch(r"[A-Za-z_]\w*", name) and name not in params and name not in aliases:
                aliases[name] = f"({target_norm})"
            m = re.fullmatch(r"([A-Za-z_]\w*)\s*([-+])\s*(\d+)", name)
            if m and m.group(1) not in params and m.group(1) not in aliases:
                inverse = "+" if m.group(2) == "-" else "-"
                aliases[m.group(1)] = f"(({target_norm}) {inverse} {m.group(3)})"


def _charset(what: str) -> tuple[str, Any] | None:
    """Allowed characters or tokens from a description like "lowercase English letters"."""
    what_l = what.lower()
    quoted = re.findall(r"""['"]([^'"]*)['"]""", what)
    classes = []
    for m in re.finditer(r"'(.)' (?:to|-) '(.)'", what):
        classes.append(f"{re.escape(m.group(1))}-{re.escape(m.group(2))}")
    m = re.search(r"\[\s*'(.)'\s*,\s*'(.)'\s*\]", what)
    if m:
        classes.append(f"{re.escape(m.group(1))}-{re.escape(m.group(2))}")
    if "lowercase" in what_l:
        classes.append("a-z")
    if "uppercase" in what_l:
        classes.append("A-Z")
    if re.search(r"\bdigits?\b", what_l) and not classes:
        classes.append("0-9")
    if classes:
        extra = "".join(re.escape(q) for q in quoted if len(q) == 1 and not re.search(r"'.' (?:to|-) '.'", what))
        return "chars", re.compile(f"[{''.join(classes)}{extra}]*")
    if quoted and all(len(q) == 1 for q in quoted):
        return "chars", re.compile(f"[{''.join(re.escape(q) for q in quoted)}]*")
    if quoted:
        return "tokens", set(quoted)
    return None


def _auto_rule(text: str, params: set[str], aliases: dict[str, str]) -> Check | None:
    body = text.rstrip(".")
    # Pure definitions (queries[i] == [li, ri]) carry no check of their own.
    if re.fullmatch(r"[A-Za-z_]\w*\[[a-z]\]\s*==?\s*\[[^\]]+\]", body):
        return lambda env: True
    # Character sets: "s and t consist only of lowercase English letters".
    m = re.fullmatch(r"(.+?) (?:consists?|contains?|only consists) (?:only |of only |of |only of )*(.+)", body)
    if m and not re.search(OPS, m.group(1)):
        allowed = _charset(m.group(2))
        if allowed and "exactly" not in m.group(2):
            kind, spec = allowed
            checks = []
            for target in _targets(m.group(1)):
                if kind == "chars":
                    checks.append(_compile(f"__ok({_normalize(target, aliases)})", params,
                                           {"__ok": lambda s, p=spec: isinstance(s, str) and p.fullmatch(s) is not None}))
                else:
                    checks.append(_compile(f"__all_in({_normalize(target, aliases)})", params,
                                           {"__all_in": lambda xs, t=spec: all(x in t for x in xs)}))
            return lambda env: all(c(env) for c in checks)
    m = re.fullmatch(r"(.+?) is (?:either|one of) (.+)", body)
    if m:
        options = [ast.literal_eval(o.strip()) for o in re.split(r",|\bor\b", m.group(2)) if o.strip()]
        return _compile(f"{_normalize(m.group(1), aliases)} in __options", params, {"__options": options})
    # Arithmetic properties.
    m = re.fullmatch(r"(.+?) is even", body)
    if m:
        return _compile(f"{_normalize(m.group(1), aliases)} % 2 == 0", params)
    m = re.fullmatch(r"(.+?) is (?:a multiple of|divisible by) (.+)", body)
    if m:
        return _compile(f"{_normalize(m.group(1), aliases)} % ({_normalize(m.group(2), aliases)}) == 0", params)
    m = re.fullmatch(r"(.+?) is a prime number", body)
    if m:
        return _compile(f"__prime({_normalize(m.group(1), aliases)})", params, {"__prime": _is_prime})
    m = re.fullmatch(r"(.+?) (?:has no|does not contain) leading zeros", body)
    if m:
        return _compile(f"{_normalize(m.group(1), aliases)}[:1] != '0'", params)
    m = re.fullmatch(r"(.+?) is sorted in (?:ascending|non-decreasing) order", body)
    if m:
        target = _normalize(m.group(1), aliases)
        return _compile(f"list({target}) == sorted({target})", params, {"list": list})
    # Uniqueness: "All queries[i] are unique", "All the edges are distinct".
    m = re.fullmatch(r"All (?:the )?(?:(?:values|elements|strings|integers|numbers) (?:of|in) )?([A-Za-z_]\w*)(?:\[i\])? are (?:unique|distinct)", body)
    if m and m.group(1) in params:
        name = m.group(1)
        return lambda env: len({json.dumps(x) for x in env[name]}) == len(env[name])
    # Comparison chains, with comma lists: `1 <= a[i], b[i] <= 10**5`.
    parts = re.split(OPS, body)
    if len(parts) >= 3:
        operands = [_normalize(o.strip(), aliases) for o in parts[::2]]
        ops = parts[1::2]
        alternatives = [_split_top_level(o) for o in operands]
        width = max(len(a) for a in alternatives)
        if any(len(a) not in (1, width) for a in alternatives):
            return None
        checks = []
        for k in range(width):
            chosen = [a[k] if len(a) == width else a[0] for a in alternatives]
            expr = chosen[0] + "".join(f" {op} {o}" for op, o in zip(ops, chosen[1:]))
            checks.append(_compile(expr, params))
        return lambda env: all(c(env) for c in checks)
    return None


# --- Hand-written guarantees -------------------------------------------------

def _is_tree(n: int, edges: list[list[int]]) -> bool:
    if len(edges) != n - 1:
        return False
    root = list(range(n))

    def find(x: int) -> int:
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    for edge in edges:
        a, b = edge[0], edge[1]
        if not (0 <= a < n and 0 <= b < n):
            return False
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        root[ra] = rb
    return True


def _parent_tree(parent: list[int]) -> bool:
    n = len(parent)
    if not parent or parent[0] != -1 or any(not 0 <= p < n for p in parent[1:]):
        return False
    return _is_tree(n, [[i, p] for i, p in enumerate(parent) if i])


def _laminar(intervals: list[list[int]]) -> bool:
    """No two intervals cross (a < c < b < d)."""
    stack: list[int] = []
    for start, end in sorted(((q[0], q[1]) for q in intervals), key=lambda q: (q[0], -q[1])):
        while stack and stack[-1] <= start:
            stack.pop()
        if stack and end > stack[-1]:
            return False
        stack.append(end)
    return True


def _has_outlier(nums: list[int]) -> bool:
    total, count = sum(nums), Counter(nums)
    for x in count:
        rest = total - x
        if rest % 2 == 0 and count[rest // 2] > (1 if rest // 2 == x else 0):
            return True
    return False


def _snake_stays_inside(n: int, commands: list[str]) -> bool:
    i = j = 0
    step = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
    for c in commands:
        di, dj = step.get(c, (0, 0))
        i, j = i + di, j + dj
        if not (0 <= i < n and 0 <= j < n):
            return False
    return True


def _valid_date(date: str) -> bool:
    try:
        d = datetime.date.fromisoformat(date)
    except ValueError:
        return False
    return len(date) == 10 and datetime.date(1900, 1, 1) <= d <= datetime.date(2100, 12, 31)


def _is_subsequence(pattern: str, source: str) -> bool:
    it = iter(source)
    return all(ch in it for ch in pattern)


def _no_overlap(rectangles: list[list[int]]) -> bool:
    if len(rectangles) > 5000:
        return True  # quadratic check skipped for very large inputs
    return not any(
        max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])
        for a, b in itertools.combinations(rectangles, 2)
    )


def _segments_disjoint(coins: list[list[int]]) -> bool:
    segments = sorted((c[0], c[1]) for c in coins)
    return all(a[1] < b[0] for a, b in zip(segments, segments[1:]))


def _distinct(values) -> bool:
    values = [json.dumps(v) for v in values]
    return len(set(values)) == len(values)


def _tree_edges(key: str, n_key: str | None = None) -> Check:
    return lambda env: _is_tree(len(env[n_key]) if n_key else len(env[key]) + 1, env[key])


# Rule text (as normalized by constraint_items) -> check, per question.
MANUAL_CHECKS: dict[int, dict[str, Check]] = {
    3243: {"There are no repeated roads among the queries.": lambda e: _distinct(e["queries"])},
    3244: {
        "There are no repeated roads among the queries.": lambda e: _distinct(e["queries"]),
        "There are no two queries such that i != j and queries[i][0] < queries[j][0] < queries[i][1] < queries[j][1].":
            lambda e: _laminar(e["queries"]),
    },
    3248: {"The input is generated such the snake will not move outside of the boundaries.":
           lambda e: _snake_stays_inside(e["n"], e["commands"])},
    3249: {"The input is generated such that edges represents a valid tree.": _tree_edges("edges")},
    3280: {
        "date[4] == date[7] == '-', and all other date[i]'s are digits.":
            lambda e: re.fullmatch(r"\d{4}-\d{2}-\d{2}", e["date"]) is not None,
        "The input is generated such that date represents a valid Gregorian calendar date between Jan 1**(st), 1900 and Dec 31**(st), 2100 (both inclusive).":
            lambda e: _valid_date(e["date"]),
    },
    3283: {
        "All positions[i] are unique.": lambda e: _distinct(e["positions"]),
        "The input is generated such that positions[i] != [kx, ky] for all 0 <= i < positions.length.":
            lambda e: [e["kx"], e["ky"]] not in e["positions"],
    },
    3291: {"The input is generated such that sum(words[i].length) <= 10**(5).": lambda e: sum(map(len, e["words"])) <= 10**5},
    3292: {"The input is generated such that sum(words[i].length) <= 10**(5).": lambda e: sum(map(len, e["words"])) <= 10**5},
    3307: {"The input is generated such that word has at least k characters after all operations.":
           lambda e: 2 ** len(e["operations"]) >= e["k"]},
    3311: {"All the edges are distinct.": lambda e: _distinct(sorted(x) for x in e["edges"])},
    3316: {
        "The input is generated such that targetIndices contains distinct elements in the range [0, n - 1].":
            lambda e: _distinct(e["targetIndices"]) and all(0 <= t < len(e["source"]) for t in e["targetIndices"]),
        "The input is generated such that pattern appears as a subsequence in source.":
            lambda e: _is_subsequence(e["pattern"], e["source"]),
    },
    3327: {"0 <= parent[i] <= n - 1 for all i >= 1.": lambda e: _parent_tree(e["parent"]),
           "parent represents a valid tree.": lambda e: _parent_tree(e["parent"])},
    3331: {"0 <= parent[i] <= n - 1 for all i >= 1.": lambda e: _parent_tree(e["parent"]),
           "parent represents a valid tree.": lambda e: _parent_tree(e["parent"])},
    3332: {
        "n == travelScore.length == travelScore[i].length == stayScore[i].length":
            lambda e: len(e["travelScore"]) == e["n"] and all(len(r) == e["n"] for r in e["travelScore"] + e["stayScore"]),
        "travelScore[i][i] == 0": lambda e: all(r[i] == 0 for i, r in enumerate(e["travelScore"]) if i < len(r)),
    },
    3354: {"There is at least one element i where nums[i] == 0.": lambda e: 0 in e["nums"]},
    3365: {"The input is generated such that s and t are anagrams of each other.": lambda e: sorted(e["s"]) == sorted(e["t"])},
    3367: {"The input is generated such that edges form a valid tree.": _tree_edges("edges")},
    3371: {"The input is generated such that at least one potential outlier exists in nums.": lambda e: _has_outlier(e["nums"])},
    3372: {
        "edges1[i].length == edges2[i].length == 2": lambda e: all(len(x) == 2 for x in e["edges1"] + e["edges2"]),
        "The input is generated such that edges1 and edges2 represent valid trees.":
            lambda e: _is_tree(len(e["edges1"]) + 1, e["edges1"]) and _is_tree(len(e["edges2"]) + 1, e["edges2"]),
    },
    3373: {
        "edges1[i].length == edges2[i].length == 2": lambda e: all(len(x) == 2 for x in e["edges1"] + e["edges2"]),
        "The input is generated such that edges1 and edges2 represent valid trees.":
            lambda e: _is_tree(len(e["edges1"]) + 1, e["edges1"]) and _is_tree(len(e["edges2"]) + 1, e["edges2"]),
    },
    3377: {"n and m consist of the same number of digits.": lambda e: len(str(e["n"])) == len(str(e["m"]))},
    3380: {"All the given points are unique.": lambda e: _distinct(e["points"])},
    3382: {"All the given points are unique.": lambda e: _distinct(zip(e["xCoord"], e["yCoord"]))},
    3386: {"The input is generated such that events is sorted in increasing order of timei.":
           lambda e: all(a[1] <= b[1] for a, b in zip(e["events"], e["events"][1:]))},
    3387: {"startCurrencyi and targetCurrencyi consist only of uppercase English letters.":
           lambda e: all(re.fullmatch(r"[A-Z]+", c) for p in e["pairs1"] + e["pairs2"] for c in p)},
    3394: {"No two rectangles overlap.": lambda e: _no_overlap(e["rectangles"])},
    3413: {"The given segments are non-overlapping.": lambda e: _segments_disjoint(e["coins"])},
    3419: {"There may be multiple edges between a pair of nodes, but they must have unique weights.":
           lambda e: _distinct(e["edges"])},
    3425: {"The input is generated such that edges represents a valid tree.": _tree_edges("edges", "nums")},
    3433: {"events[i][0] will be one of MESSAGE or OFFLINE.": lambda e: all(x[0] in ("MESSAGE", "OFFLINE") for x in e["events"])},
    3435: {"All strings in words will altogether be composed of no more than 16 unique lowercase letters.":
           lambda e: len(set("".join(e["words"]))) <= 16 and all(re.fullmatch(r"[a-z]*", w) for w in e["words"])},
    3439: {"endTime[i] <= startTime[i + 1] where i lies in the range [0, n - 2].":
           lambda e: all(e["endTime"][i] <= e["startTime"][i + 1] for i in range(len(e["startTime"]) - 1))},
    3407: {"p contains only lowercase English letters and exactly one '*'":
           lambda e: re.fullmatch(r"[a-z]*\*[a-z]*", e["p"]) is not None},
    3455: {"p contains only lowercase English letters and exactly two '*'.":
           lambda e: re.fullmatch(r"[a-z]*\*[a-z]*\*[a-z]*", e["p"]) is not None},
    3440: {"endTime[i] <= startTime[i + 1] where i lies in the range [0, n - 2].":
           lambda e: all(e["endTime"][i] <= e["startTime"][i + 1] for i in range(len(e["startTime"]) - 1))},
    3442: {"s contains at least one character with an odd frequency and one with an even frequency.":
           lambda e: any(v % 2 for v in Counter(e["s"]).values()) and any(v % 2 == 0 for v in Counter(e["s"]).values())},
    3453: {"The total area of all the squares will not exceed 10**(12).": lambda e: sum(s[2] ** 2 for s in e["squares"]) <= 10**12},
    3454: {"The total area of all the squares will not exceed 10**(15).": lambda e: sum(s[2] ** 2 for s in e["squares"]) <= 10**15},
    3464: {"The input is generated such that: points[i] lies on the boundary of the square.":
           lambda e: all(0 <= x <= e["side"] and 0 <= y <= e["side"] and (x in (0, e["side"]) or y in (0, e["side"]))
                         for x, y in e["points"])},
    3485: {"The sum of words[i].length is smaller than or equal 10**(5).": lambda e: sum(map(len, e["words"])) <= 10**5},
    3486: {"The input is generated such that edges represents a valid tree.": _tree_edges("edges", "nums")},
}

# Guarantees that need (part of) the solution itself to verify.
UNCHECKABLE: dict[int, set[str]] = {
    3311: {"The input is generated such that edges can form a 2D grid that satisfies the conditions."},
    3387: {"The input is generated such that there are no contradictions or cycles in the conversion graphs for either day.",
           "The input is generated such that the output is at most 5 * 10**(10)."},
    3433: {"The number of id<number> mentions in any \"MESSAGE\" event is between 1 and 100.",
           "0 <= <number> <= numberOfUsers - 1",
           "It is guaranteed that the user id referenced in the OFFLINE event is online at the time the event occurs."},
    3445: {"The input is generated that at least one substring has a character with an even frequency and a character with an odd frequency."},
}


# --- Audit -------------------------------------------------------------------

def build_checks(qid: int, content: str, params: set[str]) -> tuple[list[tuple[str, Check]], list[str], list[str]]:
    """(checks, uncheckable, unparsed) for one problem's Constraints."""
    items = constraint_items(content)
    aliases = element_names(content, params)
    for text in items:
        _learn_definitions(text, params, aliases)
    manual = MANUAL_CHECKS.get(qid, {})
    checks, uncheckable, unparsed = [], [], []
    for text in items:
        if text in manual:
            checks.append((text, manual[text]))
        elif text in UNCHECKABLE.get(qid, ()):
            uncheckable.append(text)
        else:
            try:
                check = _auto_rule(text, params, aliases)
            except (SyntaxError, ValueError):
                check = None
            if check is None:
                unparsed.append(text)
            else:
                checks.append((text, check))
    return checks, uncheckable, unparsed


def test_cases(problem: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """(assert source, arguments by canonical name) for each readable test case."""
    names = canonical_names(problem)
    source = problem["canonical_tests"]["source"]
    cases = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Call) and getattr(node.test.left.func, "id", "") == "candidate"):
            continue
        call = node.test.left
        try:
            values = [_test_value(a) for a in call.args]
            keywords = {k.arg: _test_value(k.value) for k in call.keywords}
        except (ValueError, TypeError, SyntaxError, RecursionError):
            continue
        env = dict(zip(names, values))
        env.update(keywords)
        cases.append((ast.unparse(node), env))
    return cases


def audit_problem(problem: dict[str, Any], content: str) -> dict[str, Any]:
    qid = int(problem["question_id"])
    checks, uncheckable, unparsed = build_checks(qid, content, set(canonical_names(problem)))
    cases = test_cases(problem)
    broken_rules: dict[str, str] = {}
    invalid: dict[str, list[str]] = {}
    for text, check in checks:
        failing = []
        for test, env in cases:
            try:
                if not check(env):
                    failing.append(test)
            except Exception as exc:  # the rule does not fit these inputs: review by hand
                broken_rules[text] = f"{type(exc).__name__}: {exc}"[:160]
                failing = []
                break
        for test in failing:
            invalid.setdefault(test, []).append(text)
    return {
        "question_id": qid,
        "constraints": constraint_items(content),
        "tests": len(cases),
        "invalid_tests": [{"test": t, "violates": r} for t, r in invalid.items()],
        "uncheckable": uncheckable,
        "unchecked": unparsed + list(broken_rules),
        "check_errors": broken_rules,
    }


def fetch_content(qids: list[int], content_dir: Path) -> None:
    """Cache each problem's LeetCode page HTML (for its exact Constraints)."""
    from .fetch_leetcode_cache import PROBLEM_LIST_CACHE, fetch_question

    content_dir.mkdir(parents=True, exist_ok=True)
    slugs = {int(r["questionFrontendId"]): r["titleSlug"] for r in json.loads(PROBLEM_LIST_CACHE.read_text())}
    for qid in qids:
        path = content_dir / f"{qid}.json"
        if path.exists():
            continue
        question = fetch_question(slugs[qid])
        path.write_text(json.dumps({"question_id": qid, "slug": slugs[qid], "content": question["content"]}))
        time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", default="b", help="Split manifest to audit (default: b).")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--content-dir", type=Path, default=DEFAULT_CONTENT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fetch", action="store_true", help="Fetch missing LeetCode pages first.")
    args = parser.parse_args()

    wanted = {int(q["question_id"]) for q in json.loads((SPLIT_DIR / f"{args.set}.json").read_text())["questions"]}
    problems = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    problems = sorted((p for p in problems if int(p["question_id"]) in wanted), key=lambda p: int(p["question_id"]))
    if args.fetch:
        fetch_content([int(p["question_id"]) for p in problems], args.content_dir)

    report = []
    for problem in problems:
        page = args.content_dir / f"{problem['question_id']}.json"
        if not page.exists():
            raise SystemExit(f"Missing LeetCode page for {problem['question_id']}; rerun with --fetch")
        report.append(audit_problem(problem, json.loads(page.read_text())["content"]))
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    affected = [r for r in report if r["invalid_tests"]]
    print(f"{len(report)} problems audited; {len(affected)} have invalid tests; "
          f"{sum(len(r['invalid_tests']) for r in report)} of {sum(r['tests'] for r in report)} test cases invalid")
    unchecked = [(r["question_id"], u) for r in report for u in r["unchecked"]]
    print(f"{sum(len(r['uncheckable']) for r in report)} rules documented as uncheckable; {len(unchecked)} unchecked:")
    for qid, rule in unchecked:
        print(f"  {qid}: {rule}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
