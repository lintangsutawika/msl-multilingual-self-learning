from __future__ import annotations

from collections import Counter

from msl_multilingual_self_learning.benchmark.leetcode.dataset import (
    load_test_split,
)
from msl_multilingual_self_learning.benchmark.leetcode.interfaces.resolve import (
    resolve_interface,
)


LANGUAGES = [
    "python",
    "cpp",
    "go",
    "java",
]


def main() -> None:
    problems = load_test_split()

    successes = Counter()
    sources = Counter()
    failures = []

    for problem in problems:
        for language in LANGUAGES:
            try:
                interface = resolve_interface(
                    problem,
                    language,
                )

                successes[language] += 1
                sources[
                    (language, interface.source)
                ] += 1

            except Exception as exc:
                failures.append(
                    (
                        problem.question_id,
                        problem.task_id,
                        language,
                        repr(exc),
                    )
                )

    expected = len(problems) * len(LANGUAGES)
    resolved = sum(successes.values())

    print(f"Problems: {len(problems)}")
    print(f"Languages: {len(LANGUAGES)}")
    print(f"Expected interfaces: {expected}")
    print(f"Resolved interfaces: {resolved}")
    print(f"Failures: {len(failures)}")

    print()
    print("Coverage by language:")

    for language in LANGUAGES:
        count = successes[language]

        print(
            f"{language:8} "
            f"{count:3}/{len(problems)} "
            f"({count / len(problems):.1%})"
        )

    print()
    print("Sources:")

    for language in LANGUAGES:
        doocs_count = sources[
            (language, "doocs")
        ]
        derived_count = sources[
            (language, "derived")
        ]

        print(
            f"{language:8} "
            f"doocs={doocs_count:3} "
            f"derived={derived_count:3}"
        )

    if failures:
        print()
        print("Failures:")

        for failure in failures:
            print(failure)


if __name__ == "__main__":
    main()