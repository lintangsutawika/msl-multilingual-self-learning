from __future__ import annotations

from collections import Counter

from .dataset import load_test_split
from .interfaces.resolve import (
    SUPPORTED_LANGUAGES,
    resolve_interface,
)


EXECUTOR_PARAMETER_TYPES = {
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


EXECUTOR_RETURN_TYPES = {
    "python": EXECUTOR_PARAMETER_TYPES["python"],
    "cpp": EXECUTOR_PARAMETER_TYPES["cpp"],
    "go": EXECUTOR_PARAMETER_TYPES["go"],
    "java": EXECUTOR_PARAMETER_TYPES["java"],
}


def main() -> None:
    problems = load_test_split()

    missing_parameters = []
    missing_returns = []

    parameter_counts = {
        language: Counter()
        for language in SUPPORTED_LANGUAGES
    }

    return_counts = {
        language: Counter()
        for language in SUPPORTED_LANGUAGES
    }

    for problem in problems:
        for language in SUPPORTED_LANGUAGES:
            interface = resolve_interface(
                problem,
                language,
            )

            for parameter in interface.parameters:
                parameter_counts[
                    language
                ][parameter.type] += 1

                if (
                    parameter.type
                    not in EXECUTOR_PARAMETER_TYPES[
                        language
                    ]
                ):
                    missing_parameters.append(
                        (
                            problem.task_id,
                            language,
                            parameter.name,
                            parameter.type,
                        )
                    )

            return_counts[
                language
            ][interface.return_type] += 1

            if (
                interface.return_type
                not in EXECUTOR_RETURN_TYPES[
                    language
                ]
            ):
                missing_returns.append(
                    (
                        problem.task_id,
                        language,
                        interface.return_type,
                    )
                )

    print()
    print("=" * 80)
    print("EXECUTOR TYPE COMPATIBILITY AUDIT")
    print("=" * 80)

    print(
        f"Problems: {len(problems)}"
    )
    print(
        f"Interfaces: "
        f"{len(problems) * len(SUPPORTED_LANGUAGES)}"
    )

    print()
    print(
        "Missing parameter types:",
        len(missing_parameters),
    )

    for item in missing_parameters[:50]:
        print(" ", item)

    print()
    print(
        "Missing return types:",
        len(missing_returns),
    )

    for item in missing_returns[:50]:
        print(" ", item)

    success = (
        not missing_parameters
        and not missing_returns
    )

    print()

    if success:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()