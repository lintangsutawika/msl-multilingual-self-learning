from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .dataset import load_test_split
from .interfaces.resolve import resolve_interface
from .interfaces.derive import parse_hf_python_signature


LANGUAGES = (
    "python",
    "cpp",
    "go",
    "java",
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


def _determine_type_source(
    language: str,
    interface: Any,
) -> str:
    raw_signature = interface.raw_signature

    if not raw_signature:
        return "unknown"

    for parameter in interface.parameters:
        if not _parameter_type_appears_in_signature(
            language,
            raw_signature,
            parameter.name,
            parameter.type,
        ):
            return "leetcode_starter"

    if not _return_type_appears_in_signature(
        language,
        raw_signature,
        interface.callable_name,
        interface.return_type,
    ):
        return "leetcode_starter"

    return "doocs"


def _interface_to_dict(
    language: str,
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
            for parameter
            in interface.parameters
        ],
        "return_type": interface.return_type,
        "signature_source": "doocs",
        "type_source": _determine_type_source(
            language,
            interface,
        ),
    }


def _resolve_all_interfaces(
    problem: Any,
) -> dict[str, Any] | None:
    interfaces = {}

    for language in LANGUAGES:
        interface = resolve_interface(
            problem,
            language,
        )

        if interface.source != "doocs":
            return None

        if not interface.raw_signature:
            return None

        interfaces[language] = (
            _interface_to_dict(
                language,
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
                "doocs/leetcode"
            ),
        },
    }


def generate_dataset(
    output_path: Path,
) -> None:
    problems = load_test_split()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []
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

    # Only create/replace the output after generation
    # succeeds. This prevents leaving behind an empty
    # dataset if something crashes midway.
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for record in records:
            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            output_file.write(
                "\n"
            )

    print(
        f"HF test problems: {len(problems)}"
    )

    print(
        f"Languages required: "
        f"{len(LANGUAGES)}"
    )

    print(
        f"Problems kept: "
        f"{len(records)}"
    )

    print(
        f"Problems skipped: "
        f"{len(skipped)}"
    )

    print(
        f"Output: {output_path}"
    )

    if skipped:
        print(
            "\nSkipped problems:"
        )

        for question_id, task_id in skipped:
            print(
                f"  {question_id} "
                f"{task_id}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a multilingual LeetCode "
            "benchmark using only problems with "
            "native Doocs signatures for Python, "
            "C++, Go, and Java."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Path to the output JSONL file. "
            f"Default: {DEFAULT_OUTPUT}"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    generate_dataset(
        args.output
    )


if __name__ == "__main__":
    main()
