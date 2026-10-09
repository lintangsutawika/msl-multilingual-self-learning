"""Decide which canonical tests are fair for every language.

A test is dropped when its input breaks the problem's Constraints (each rule on
the LeetCode page becomes a check; MANUAL_CHECKS covers "The input is generated
such that ..." rules, UNCHECKABLE lists rules that need the solution), when a
value does not fit a declared type in some language, or when it expects inf/nan.
"""
from __future__ import annotations

import ast
import datetime
import html
import itertools
import json
import math
import re
from collections import Counter
from typing import Any, Callable

Check = Callable[[dict[str, Any]], bool]
OPS = r"(<=|>=|==|!=|<|>)"
SAFE_BUILTINS = {
    "len": len, "all": all, "any": any, "range": range, "set": set, "sorted": sorted,
    "min": min, "max": max, "sum": sum, "abs": abs, "int": int, "floor": math.floor,
}


# --- Test values and types -----------------------------------------------

def canonical_names(problem: dict[str, Any]) -> list[str]:
    """Canonical parameter names for the shared Python judge."""
    names = problem.get("metadata", {}).get("canonical_parameter_names")
    if names is None:
        names = [p["name"] for p in problem["interfaces"]["python"]["parameters"]]
        for node in ast.walk(ast.parse(problem["canonical_tests"]["source"])):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "candidate":
                if any(k.arg not in names for k in node.keywords):
                    raise ValueError(
                        "HF/Doocs parameter names differ; regenerate dataset to retain canonical_parameter_names"
                    )
    return names


# A problem needs this many fair test cases, or a wrong solution can pass by luck.
MIN_VALID_TESTS = 10

INT32_MAX = 2**31 - 1
INT64_MAX = 2**63 - 1
JS_SAFE_MAX = 2**53 - 1


def integer_limit(language: str, type_name: str) -> int | None:
    """Largest integer a declared LeetCode type holds exactly (None: unbounded or not an integer)."""
    if language in ("python", "ruby"):
        return None
    if language in ("javascript", "typescript"):
        return JS_SAFE_MAX
    if language == "php":
        return INT64_MAX
    if language == "go":
        return INT64_MAX if re.search(r"\bint(64)?\b", type_name) else None
    if language == "rust":
        return INT64_MAX if "i64" in type_name else INT32_MAX if "i32" in type_name else None
    if re.search(r"\blong\b", type_name):
        return INT64_MAX
    if re.search(r"\b(int|Integer)\b", type_name):
        return INT32_MAX
    return None


def _integers(value: Any):
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _integers(item)


def _non_finite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    return isinstance(value, (list, tuple)) and any(_non_finite(item) for item in value)


def _unrepresentable_languages(problem: dict[str, Any], args: list[Any], expected: Any) -> list[str]:
    """Languages whose declared parameter/return types cannot hold this test case."""
    if _non_finite(args) or _non_finite(expected):
        # inf/nan cannot cross the JSON transport, so no language can pass.
        return list(problem["interfaces"])
    languages = []
    for language, interface in problem["interfaces"].items():
        pairs = list(zip((p["type"] for p in interface["parameters"]), args))
        pairs.append((interface["return_type"], expected))
        for type_name, value in pairs:
            limit = integer_limit(language, type_name)
            if limit is not None and any(not -limit - 1 <= x <= limit for x in _integers(value)):
                languages.append(language)
                break
    return languages


_ARITHMETIC_NODES = (
    ast.Expression, ast.Constant, ast.List, ast.Tuple, ast.Load,
    ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Pow, ast.USub, ast.UAdd,
)
# Names the reference solution's outputs may contain (the judge star-imports math).
_TEST_NAMES = {"inf": math.inf, "nan": math.nan}


def _test_value(node: ast.AST) -> Any:
    """Value of a test argument: a literal, or plain arithmetic such as 10**9, [0] * n or -inf."""
    try:
        return ast.literal_eval(node)
    except ValueError:
        expression = ast.Expression(node)
        for n in ast.walk(expression):
            if not isinstance(n, _ARITHMETIC_NODES) and not (isinstance(n, ast.Name) and n.id in _TEST_NAMES):
                raise
        return eval(compile(expression, "<test>", "eval"), {"__builtins__": {}, **_TEST_NAMES})


def _plain_value(value: Any) -> bool:
    """Whether the JSON transport can carry the value to every language."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    return isinstance(value, (list, tuple)) and all(_plain_value(v) for v in value)


def _has_decimal(value: Any, whole_ok: bool = False) -> bool:
    """A finite float anywhere in the value (inf/nan are reported as non-finite). With
    whole_ok, exact whole numbers such as 59048.0 do not count."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return False
        return not (whole_ok and value.is_integer() and abs(value) <= 2**53)
    return isinstance(value, (list, tuple)) and any(_has_decimal(v, whole_ok) for v in value)


def _decimal_for_integer(problem: dict[str, Any], args: list[Any], expected: Any) -> bool:
    """Whether a test passes a decimal where LeetCode declares integers (Python types, e.g.
    int or List[List[int]]): Go and Rust cannot even parse it, others round it. An expected
    answer may be a whole-number float (59048.0 == 59048 in the Python comparison), but not
    10.5, which no integer answer can equal."""
    python = problem.get("interfaces", {}).get("python")
    if not python:
        return False
    integer = lambda t: re.search(r"\bint\b", t) is not None and "float" not in t
    types = [p["type"] for p in python["parameters"]]
    if any(integer(t) and _has_decimal(v) for t, v in zip(types, args)):
        return True
    return integer(python["return_type"]) and _has_decimal(expected, whole_ok=True)


def filter_canonical_tests(
    problem: dict[str, Any],
    invalid: dict[str, list[str]] | None = None,
) -> tuple[str, list[dict[str, Any]], int]:
    """Drop tests that do not fit a declared type in some language, pass a decimal where an
    integer is declared, expect inf/nan, or are listed in `invalid` (assert source -> violated rules).

    Returns (source, dropped, number of test cases kept).
    """
    invalid = invalid or {}
    source = problem["canonical_tests"]["source"]
    names = canonical_names(problem)
    asserts = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assert)
        and isinstance(node.test, ast.Compare)
        and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)
        and isinstance(node.test.left, ast.Call)
        and isinstance(node.test.left.func, ast.Name) and node.test.left.func.id == "candidate"
    ]
    dropped = []
    for node in asserts:
        test = ast.unparse(node)
        if test in invalid:
            dropped.append({"line": node.lineno, "end_line": node.end_lineno, "violates": invalid[test]})
            continue
        call = node.test.left
        try:
            values = [_test_value(a) for a in call.args]
            keywords = {k.arg: _test_value(k.value) for k in call.keywords}
            expected = _test_value(node.test.comparators[0])
        except (ValueError, TypeError, SyntaxError, RecursionError):
            values = None
        if values is not None:
            values += [keywords[n] for n in names[len(values):] if n in keywords]
        if values is None or len(values) != len(names) or not _plain_value([values, expected]):
            # Not a plain value (e.g. a literal `...` copied from an abbreviated example): unchecked, so dropped.
            dropped.append({"line": node.lineno, "end_line": node.end_lineno, "unreadable": True})
            continue
        if _decimal_for_integer(problem, values, expected):
            dropped.append({"line": node.lineno, "end_line": node.end_lineno, "decimal_in_integer": True})
            continue
        languages = _unrepresentable_languages(problem, values, expected)
        if languages:
            dropped.append({"line": node.lineno, "end_line": node.end_lineno, "unrepresentable_in": languages})
    kept = len(asserts) - len(dropped)
    if not dropped:
        return source, [], kept
    removed = {n for d in dropped for n in range(d["line"], d["end_line"] + 1)}
    lines = source.splitlines(keepends=True)
    return "".join(line for n, line in enumerate(lines, 1) if n not in removed), dropped, kept



# --- Page text ---------------------------------------------------------------

def _plain(fragment: str) -> str:
    """HTML fragment -> plain text, with 10<sup>5</sup> -> 10**(5) and l<sub>i</sub> -> li."""
    fragment = re.sub(r"<sup>\s*(.*?)\s*</sup>", r"**(\1)", fragment, flags=re.S)
    fragment = re.sub(r"<[^>]+>", "", fragment)
    fragment = html.unescape(fragment).replace(" ", " ").replace("−", "-")
    # Some pages write 10^9 as plain text; in a rule it is a power, not XOR.
    fragment = re.sub(r"(?<=[\w)])\s*\^\s*(?=[\w(-])", "**", fragment)
    return " ".join(fragment.split())


def constraint_items(content: str) -> list[str]:
    """Plain-text items of the page's Constraints list."""
    m = re.search(r"Constraints:(.*?)(</ul>|$)", content, re.S)
    if not m:
        return []
    return [_plain(item) for item in re.findall(r"<li\b[^>]*>(.*?)</li>", m.group(1), re.S)]


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
        before = max([int(k) for k in re.findall(rf"\[\s*{var}\s*-\s*(\d+)\s*\]", expr)], default=0)
        after = max([int(k) for k in re.findall(rf"\[\s*{var}\s*\+\s*(\d+)\s*\]", expr)], default=0)
        body = f"all(({body}) for {var} in range({before}, len({seq}) - {after}))"
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


# Words for characters that rules name instead of quoting.
NAMED_CHARS = {
    "space": " ", "spaces": " ", "whitespace": " ", "parentheses": "()", "brackets": "[]", "braces": "{}",
    "period": ".", "periods": ".", "dot": ".", "dots": ".", "slash": "/", "slashes": "/", "underscore": "_",
    "comma": ",", "commas": ",", "plus": "+", "minus": "-", "hyphen": "-", "hyphens": "-", "dash": "-",
    "asterisk": "*", "colon": ":", "semicolon": ";", "apostrophe": "'",
}
# Descriptions too vague to check ("symbols", "printable ASCII"): the rule is reported as unchecked.
VAGUE_CHARSET = re.compile(r"\bsymbols?\b|printable|ascii|any characters?|special characters?|punctuation|non-?alpha", re.I)


def _charset(what: str) -> Callable[[Any], bool] | None:
    """A check for "consists of ..." rules like "lowercase English letters, digits and '_'".

    None when the description is not understood.
    """
    if VAGUE_CHARSET.search(what):
        return None
    what_l = what.lower()
    quoted = re.findall(r"""'([^']*)'|"([^"]*)\"""", what)
    quoted = [a or b for a, b in quoted]
    ranges = re.findall(r"'(.)'\s*(?:to|-)\s*'(.)'", what) + re.findall(r"\[\s*'(.)'\s*,\s*'(.)'\s*\]", what)
    classes = [f"{re.escape(a)}-{re.escape(b)}" for a, b in ranges]
    # An explicit range ("digits '0' to '4'", "letters 'a' to 'e'") narrows the generic word.
    digit_range = any(a.isdigit() for a, _ in ranges)
    letter_range = any(a.isalpha() for a, _ in ranges)
    lower = re.search(r"\blower[- ]?case\b", what_l)
    upper = re.search(r"\bupper[- ]?case\b", what_l)
    if not letter_range:
        if lower:
            classes.append("a-z")
        if upper:
            classes.append("A-Z")
        if re.search(r"\bletters?\b", what_l) and not lower and not upper:
            classes.append("a-zA-Z")
    if re.search(r"\b(?:digits?|integers?|numbers?|numeric)\b", what_l) and not digit_range:
        classes.append("0-9")
    named = "".join(c for word, c in NAMED_CHARS.items() if re.search(rf"\b{word}\b", what_l))
    chars = "".join(set("".join(quoted) + named))
    if not classes and not chars:
        return None
    pattern = re.compile(f"[{''.join(classes)}{re.escape(chars)}]*" if chars else f"[{''.join(classes)}]*")
    tokens = set(quoted)

    def ok(value: Any) -> bool:
        if isinstance(value, str):
            return pattern.fullmatch(value) is not None
        return isinstance(value, list) and all(ok(x) or (isinstance(x, str) and x in tokens) for x in value)
    return ok


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
            checks = [_compile(f"__ok({_normalize(target, aliases)})", params, {"__ok": allowed})
                      for target in _targets(m.group(1))]
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
    3289: {"The input is generated such that nums contains exactly two repeated elements.":
           lambda e: sorted(Counter(e["nums"]).values()).count(2) == 2 and max(Counter(e["nums"]).values()) == 2},
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


def _passes(check: Check, env: dict[str, Any]) -> bool:
    try:
        return bool(check(env))
    except Exception:
        return False


def audit_problem(problem: dict[str, Any], content: str,
                  examples: list[dict[str, Any]] = ()) -> dict[str, Any]:
    """Tests that break a Constraints rule. `examples` are the page's public examples
    (arguments by name): a rule that rejects one is misread, so it is not used."""
    qid = int(problem["question_id"])
    checks, uncheckable, unparsed = build_checks(qid, content, set(canonical_names(problem)))
    cases = test_cases(problem)
    broken_rules: dict[str, str] = {}
    invalid: dict[str, list[str]] = {}
    for text, check in checks:
        if any(not _passes(check, env) for env in examples):
            broken_rules[text] = "rejects a public example"
            continue
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
