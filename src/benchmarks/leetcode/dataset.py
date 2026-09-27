"""Build LeetCode problem rows (the execution-dataset schema) live from HF.

Replaces the old committed ``leetcode_multilingual_*.jsonl`` + ``generate_dataset.py``:
``load_problems_hf(split)`` fetches ``newfacade/LeetCodeDataset`` rows for a split and
reconstructs the per-language ``interfaces`` (fetched lazily from LeetCode's
``codeSnippets`` API and cached locally) and ``canonical_tests`` (from HF ``test``),
returning the same problem-row dicts that ``main.py``/``adapter`` consume. No committed
JSONL, no doocs checkout, no manual cache prep: snippets are fetched on demand and
cached under ``benchmarks/leetcode/data/cache/leetcode_snippets``.
"""
from __future__ import annotations

import json
import re

from dataclasses import dataclass
from datasets import load_dataset

import requests
from pathlib import Path
from typing import Any

from .interfaces.derive import parse_hf_python_signature
from .interfaces.leetcode import (
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


def _fetch_snippets(question_id: int, title_slug: str) -> dict[str, str] | None:
    """Fetch a problem's per-language codeSnippets from LeetCode and cache them.

    Returns {langSlug: code} or None on any failure. Caches the raw snippet record
    under leetcode_snippets/<qid>.json so repeat builds don't re-hit the API.
    """
    query = {
        "query": (
            "query questionData($titleSlug: String!) { "
            "question(titleSlug: $titleSlug) { "
            "questionId title codeSnippets { langSlug code } } }"
        ),
        "variables": {"titleSlug": title_slug},
    }
    try:
        resp = requests.post(
            _GRAPHQL_URL,
            json=query,
            headers=_GRAPHQL_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        qn = (resp.json().get("data") or {}).get("question") or {}
        snippets = qn.get("codeSnippets")
        if not snippets:
            return None
    except Exception:
        return None

    record = {
        "question_id": question_id,
        "code_snippets": snippets,
    }
    out = LEETCODE_SNIPPET_CACHE / f"{question_id}.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass  # cache is best-effort; interfaces still usable this run
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


def load_problems_hf(
    split: str = "test",
) -> list[dict[str, Any]]:
    """Return problem rows for a LeetCode split (test | train).

    Mirrors the old generate_dataset output schema so main.py can consume it
    directly (no committed JSONL). Empty if the snippets cache is unpopulated.
    """
    if split == "train":
        problems = load_train_split()
    else:
        problems = load_test_split()

    records: list[dict[str, Any]] = []
    skipped = []

    for problem in problems:
        interfaces = _resolve_all_interfaces(
            problem
        )

        if interfaces is None:
            skipped.append(
                (
                    _problem_value(
                        problem,
                        "question_id",
                    ),
                    _problem_value(
                        problem,
                        "task_id",
                    ),
                )
            )
            continue

        record = _build_record(
            problem,
            interfaces,
        )

        records.append(
            record
        )

    if not records:
        raise RuntimeError(
            "No problems with complete LeetCode interfaces were generated"
        )

    return records

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


def load_train_split() -> list[LeetCodeProblem]:
    dataset = load_dataset(
        DATASET_NAME,
        split="train",
    )

    problems: list[LeetCodeProblem] = []

    for row in dataset:
        problem = LeetCodeProblem(
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
        )

        problems.append(problem)

    return problems


def load_test_split() -> list[LeetCodeProblem]:
    dataset = load_dataset(
        DATASET_NAME,
        split="test",
    )

    problems: list[LeetCodeProblem] = []

    for row in dataset:
        problem = LeetCodeProblem(
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
        )

        problems.append(problem)

    return problems


def load_problem(
    dataset_path: Path,
    question_id: int,
) -> dict[str, Any]:
    """Look up one generated JSONL record (the CLI/adapter input) by question_id."""
    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if int(record["question_id"]) == question_id:
                return record

    raise ValueError(
        f"Question ID {question_id} "
        f"not found in {dataset_path}"
    )