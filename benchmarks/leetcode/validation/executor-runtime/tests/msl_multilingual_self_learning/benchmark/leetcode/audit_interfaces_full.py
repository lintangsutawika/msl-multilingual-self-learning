from __future__ import annotations

from collections import Counter, defaultdict

from .dataset import load_test_split
from .interfaces.resolve import (
    SUPPORTED_LANGUAGES,
    resolve_interface,
)


SUPPORTED_TYPES = {
    "python": {
        "int",
        "str",
        "bool",
        "float",
        "List[int]",
        "List[str]",
        "List[bool]",
        "List[float]",
        "List[List[int]]",
        "List[List[str]]",
        "Optional[TreeNode]",
    },

    "cpp": {
        "int",
        "long long",
        "bool",
        "double",
        "char",

        "string",
        "std::string",

        "vector<int>",
        "vector<int>&",
        "vector<int> &",

        "vector<long long>",
        "vector<long long>&",

        "vector<double>",
        "vector<double>&",

        "vector<bool>",

        "vector<string>",
        "vector<string>&",

        "vector<vector<int>>",
        "vector<vector<int>>&",

        "vector<vector<string>>",
        "vector<vector<string>>&",

        "TreeNode*",
    },

    "go": {
        "int",
        "int64",
        "string",
        "bool",
        "float64",
        "byte",

        "[]int",
        "[]int64",
        "[]string",
        "[]bool",
        "[]float64",

        "[][]int",
        "[][]string",

        "*TreeNode",
    },

    "java": {
        "int",
        "long",
        "String",
        "boolean",
        "double",
        "char",

        "int[]",
        "long[]",
        "String[]",
        "boolean[]",
        "double[]",

        "int[][]",
        "String[][]",

        "List<Integer>",
        "List<String>",
        "List<List<Integer>>",
        "List<List<String>>",

        "TreeNode",
    },
}


EXPECTED_CONTAINERS = {
    "python": {"Solution"},
    "cpp": {"Solution"},
    "go": {None},
    "java": {"Solution"},
}


def main() -> None:
    problems = load_test_split()

    languages = [
        language
        for language in SUPPORTED_LANGUAGES
        if language in {
            "python",
            "cpp",
            "go",
            "java",
        }
    ]

    expected_total = (
        len(problems)
        * len(languages)
    )

    resolved_total = 0

    failures: list[str] = []
    unsupported_types: list[str] = []
    bad_containers: list[str] = []
    empty_callables: list[str] = []

    source_counts = Counter()

    parameter_type_counts = {
        language: Counter()
        for language in languages
    }

    return_type_counts = {
        language: Counter()
        for language in languages
    }

    arity_counts = {
        language: Counter()
        for language in languages
    }

    callable_counts = {
        language: Counter()
        for language in languages
    }

    problem_failures = defaultdict(
        list
    )

    for problem in problems:
        for language in languages:
            try:
                interface = resolve_interface(
                    problem,
                    language,
                )

            except Exception as exc:
                message = (
                    f"{problem.question_id} "
                    f"{problem.task_id} "
                    f"[{language}] "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                failures.append(
                    message
                )

                problem_failures[
                    problem.task_id
                ].append(
                    language
                )

                continue

            resolved_total += 1

            source_counts[
                (
                    language,
                    interface.source,
                )
            ] += 1

            arity_counts[
                language
            ][
                len(
                    interface.parameters
                )
            ] += 1

            callable_counts[
                language
            ][
                interface.callable_name
            ] += 1

            if not interface.callable_name:
                empty_callables.append(
                    (
                        f"{problem.task_id} "
                        f"[{language}]"
                    )
                )

            expected_containers = (
                EXPECTED_CONTAINERS[
                    language
                ]
            )

            if (
                interface.container
                not in expected_containers
            ):
                bad_containers.append(
                    (
                        f"{problem.task_id} "
                        f"[{language}] "
                        f"container="
                        f"{interface.container!r}"
                    )
                )

            supported_types = (
                SUPPORTED_TYPES[
                    language
                ]
            )

            for parameter in (
                interface.parameters
            ):
                parameter_type_counts[
                    language
                ][
                    parameter.type
                ] += 1

                if (
                    parameter.type
                    not in supported_types
                ):
                    unsupported_types.append(
                        (
                            f"{problem.task_id} "
                            f"[{language}] "
                            f"parameter "
                            f"{parameter.name}: "
                            f"{parameter.type}"
                        )
                    )

            return_type_counts[
                language
            ][
                interface.return_type
            ] += 1

            if (
                interface.return_type
                not in supported_types
            ):
                unsupported_types.append(
                    (
                        f"{problem.task_id} "
                        f"[{language}] "
                        f"return: "
                        f"{interface.return_type}"
                    )
                )

    print()
    print("=" * 80)
    print("FULL INTERFACE COVERAGE AUDIT")
    print("=" * 80)

    print(
        f"Problems: {len(problems)}"
    )

    print(
        f"Languages: {len(languages)}"
    )

    print(
        f"Expected interfaces: "
        f"{expected_total}"
    )

    print(
        f"Resolved interfaces: "
        f"{resolved_total}"
    )

    print(
        f"Resolution failures: "
        f"{len(failures)}"
    )

    print()

    print("Sources:")

    for language in languages:
        doocs = source_counts[
            (
                language,
                "doocs",
            )
        ]

        derived = source_counts[
            (
                language,
                "derived",
            )
        ]

        print(
            f"  {language:6s} "
            f"doocs={doocs:3d} "
            f"derived={derived:3d}"
        )

    print()

    print("Parameter types:")

    for language in languages:
        print()
        print(
            f"  [{language}]"
        )

        for type_name, count in (
            parameter_type_counts[
                language
            ].most_common()
        ):
            marker = (
                ""
                if type_name
                in SUPPORTED_TYPES[
                    language
                ]
                else "  <-- UNSUPPORTED"
            )

            print(
                f"    {type_name:30s} "
                f"{count:4d}"
                f"{marker}"
            )

    print()

    print("Return types:")

    for language in languages:
        print()
        print(
            f"  [{language}]"
        )

        for type_name, count in (
            return_type_counts[
                language
            ].most_common()
        ):
            marker = (
                ""
                if type_name
                in SUPPORTED_TYPES[
                    language
                ]
                else "  <-- UNSUPPORTED"
            )

            print(
                f"    {type_name:30s} "
                f"{count:4d}"
                f"{marker}"
            )

    print()

    print("Arities:")

    for language in languages:
        values = ", ".join(
            (
                f"{arity}: {count}"
                for arity, count
                in sorted(
                    arity_counts[
                        language
                    ].items()
                )
            )
        )

        print(
            f"  {language:6s} "
            f"{values}"
        )

    print()

    print(
        "Unsupported types: "
        f"{len(unsupported_types)}"
    )

    for item in unsupported_types[
        :50
    ]:
        print(
            f"  {item}"
        )

    if (
        len(unsupported_types)
        > 50
    ):
        print(
            "  ... "
            f"{len(unsupported_types) - 50} "
            "more"
        )

    print()

    print(
        "Bad containers: "
        f"{len(bad_containers)}"
    )

    for item in bad_containers[
        :50
    ]:
        print(
            f"  {item}"
        )

    print()

    print(
        "Empty callables: "
        f"{len(empty_callables)}"
    )

    for item in empty_callables[
        :50
    ]:
        print(
            f"  {item}"
        )

    print()

    if failures:
        print("Resolution failures:")

        for item in failures[
            :50
        ]:
            print(
                f"  {item}"
            )

        print()

    success = (
        resolved_total
        == expected_total
        and not failures
        and not unsupported_types
        and not bad_containers
        and not empty_callables
    )

    if success:
        print(
            "RESULT: PASS"
        )

    else:
        print(
            "RESULT: FAIL"
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()