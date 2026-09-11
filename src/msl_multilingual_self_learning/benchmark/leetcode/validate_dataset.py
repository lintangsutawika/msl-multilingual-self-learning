from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


LANGUAGES = (
    "python",
    "cpp",
    "go",
    "java",
)

EXPECTED_CONTAINERS = {
    "python": "Solution",
    "cpp": "Solution",
    "go": None,
    "java": "Solution",
}

VALID_TYPE_SOURCES = {
    "doocs",
    "leetcode_starter",
}

DEFAULT_DATASET = Path(
    "benchmarks/leetcode/data/leetcode_multilingual.jsonl"
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


def _name_appears(
    raw_signature: str,
    name: str,
) -> bool:
    return bool(
        re.search(
            rf"\b{re.escape(name)}\b",
            raw_signature,
        )
    )


def _parameter_type_appears(
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


def _return_type_appears(
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


def _validate_interface(
    question_id: Any,
    task_id: str,
    language: str,
    interface: dict[str, Any],
) -> list[str]:
    errors = []

    prefix = (
        f"{question_id} "
        f"{task_id} "
        f"[{language}]"
    )

    raw_signature = interface.get(
        "raw_signature"
    )

    callable_name = interface.get(
        "callable"
    )

    parameters = interface.get(
        "parameters"
    )

    return_type = interface.get(
        "return_type"
    )

    container = interface.get(
        "container"
    )

    signature_source = interface.get(
        "signature_source"
    )

    type_source = interface.get(
        "type_source"
    )

    if signature_source != "doocs":
        errors.append(
            f"{prefix}: "
            f"signature_source is "
            f"{signature_source!r}, "
            "expected 'doocs'"
        )

    if type_source not in VALID_TYPE_SOURCES:
        errors.append(
            f"{prefix}: "
            f"invalid type_source "
            f"{type_source!r}"
        )

    if (
        not isinstance(
            raw_signature,
            str,
        )
        or not raw_signature.strip()
    ):
        errors.append(
            f"{prefix}: "
            "missing raw_signature"
        )
        return errors

    if (
        not isinstance(
            callable_name,
            str,
        )
        or not callable_name.strip()
    ):
        errors.append(
            f"{prefix}: "
            "missing callable"
        )
        return errors

    if not isinstance(
        parameters,
        list,
    ):
        errors.append(
            f"{prefix}: "
            "parameters must be a list"
        )
        return errors

    if (
        not isinstance(
            return_type,
            str,
        )
        or not return_type.strip()
    ):
        errors.append(
            f"{prefix}: "
            "missing return_type"
        )

    expected_container = (
        EXPECTED_CONTAINERS[
            language
        ]
    )

    if container != expected_container:
        errors.append(
            f"{prefix}: "
            f"container is "
            f"{container!r}, "
            f"expected "
            f"{expected_container!r}"
        )

    if not _name_appears(
        raw_signature,
        callable_name,
    ):
        errors.append(
            f"{prefix}: "
            f"callable "
            f"{callable_name!r} "
            "does not appear in "
            "raw_signature"
        )

    doocs_types_complete = True

    for parameter in parameters:
        name = parameter.get(
            "name"
        )

        type_name = parameter.get(
            "type"
        )

        if (
            not isinstance(
                name,
                str,
            )
            or not name.strip()
        ):
            errors.append(
                f"{prefix}: "
                "parameter missing name"
            )
            continue

        if (
            not isinstance(
                type_name,
                str,
            )
            or not type_name.strip()
        ):
            errors.append(
                f"{prefix}: "
                f"parameter "
                f"{name!r} "
                "missing type"
            )
            continue

        if not _name_appears(
            raw_signature,
            name,
        ):
            errors.append(
                f"{prefix}: "
                f"parameter name "
                f"{name!r} "
                "does not appear in "
                "raw_signature"
            )

        if not _parameter_type_appears(
            language,
            raw_signature,
            name,
            type_name,
        ):
            doocs_types_complete = False

    if return_type:
        if not _return_type_appears(
            language,
            raw_signature,
            callable_name,
            return_type,
        ):
            doocs_types_complete = False

    expected_type_source = (
        "doocs"
        if doocs_types_complete
        else "leetcode_starter"
    )

    if (
        type_source
        != expected_type_source
    ):
        errors.append(
            f"{prefix}: "
            f"type_source is "
            f"{type_source!r}, "
            f"but signature implies "
            f"{expected_type_source!r}"
        )

    return errors


def validate_dataset(
    dataset_path: Path,
) -> int:
    errors = []

    language_counts = Counter()
    type_source_counts = Counter()

    question_ids = set()

    records = 0

    if not dataset_path.exists():
        print(
            f"Dataset does not exist: "
            f"{dataset_path}"
        )
        print(
            "RESULT: FAIL"
        )
        return 1

    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as dataset_file:
        for line_number, line in enumerate(
            dataset_file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                errors.append(
                    f"line {line_number}: "
                    f"invalid JSON: {exc}"
                )
                continue

            records += 1

            question_id = record.get(
                "question_id"
            )

            task_id = record.get(
                "task_id"
            )

            if question_id in question_ids:
                errors.append(
                    f"{question_id} "
                    f"{task_id}: "
                    "duplicate question_id"
                )
            else:
                question_ids.add(
                    question_id
                )

            interfaces = record.get(
                "interfaces"
            )

            if not isinstance(
                interfaces,
                dict,
            ):
                errors.append(
                    f"{question_id} "
                    f"{task_id}: "
                    "missing interfaces object"
                )
                continue

            actual_languages = set(
                interfaces
            )

            expected_languages = set(
                LANGUAGES
            )

            if (
                actual_languages
                != expected_languages
            ):
                errors.append(
                    f"{question_id} "
                    f"{task_id}: "
                    "languages are "
                    f"{sorted(actual_languages)}, "
                    "expected "
                    f"{sorted(expected_languages)}"
                )

            for language in LANGUAGES:
                interface = interfaces.get(language)
                if not isinstance(interface, dict):
                    errors.append(
                        f"{question_id} {task_id}: "
                        f"missing {language} interface object"
                    )
                    continue

                language_counts[
                    language
                ] += 1

                type_source_counts[
                    (
                        language,
                        interface.get(
                            "type_source"
                        ),
                    )
                ] += 1

                errors.extend(
                    _validate_interface(
                        question_id,
                        task_id,
                        language,
                        interface,
                    )
                )

    if records == 0:
        errors.append(
            "dataset contains zero problems"
        )

    print(
        "=" * 72
    )

    print(
        "LEETCODE MULTILINGUAL DATASET VALIDATION"
    )

    print(
        "=" * 72
    )

    print(
        f"Dataset: {dataset_path}"
    )

    print(
        f"Problems: {records}"
    )

    print(
        f"Unique question IDs: "
        f"{len(question_ids)}"
    )

    print()

    print(
        "Interfaces:"
    )

    for language in LANGUAGES:
        print(
            f"  {language:<7} "
            f"{language_counts[language]}"
        )

    print()

    print(
        "Type provenance:"
    )

    for language in LANGUAGES:
        doocs_count = (
            type_source_counts[
                (
                    language,
                    "doocs",
                )
            ]
        )

        fallback_count = (
            type_source_counts[
                (
                    language,
                    "leetcode_starter",
                )
            ]
        )

        print(
            f"  {language:<7} "
            f"doocs={doocs_count} "
            f"leetcode_starter="
            f"{fallback_count}"
        )

    print()

    if errors:
        print(
            f"Errors: {len(errors)}"
        )

        for error in errors:
            print(
                f"  {error}"
            )

        print()

        print(
            "RESULT: FAIL"
        )

        return 1

    print(
        "Errors: 0"
    )

    print()

    print(
        "RESULT: PASS"
    )

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the generated "
            "multilingual LeetCode JSONL "
            "dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=(
            "Path to the generated "
            "JSONL dataset."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raise SystemExit(
        validate_dataset(
            args.dataset
        )
    )


if __name__ == "__main__":
    main()