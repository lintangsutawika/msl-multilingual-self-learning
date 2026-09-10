from __future__ import annotations

import ast
from collections import Counter

from .dataset import load_test_split
from .interfaces.resolve import resolve_interface


def _candidate_calls(
    test_source: str,
) -> list[ast.Call]:
    tree = ast.parse(
        test_source
    )

    calls: list[ast.Call] = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        if not isinstance(
            node.func,
            ast.Name,
        ):
            continue

        if node.func.id != "candidate":
            continue

        calls.append(node)

    return calls


def main() -> None:
    problems = load_test_split()

    total_calls = 0

    positional_calls = 0
    keyword_calls = 0
    mixed_calls = 0

    arities = Counter()

    failures: list[str] = []

    mismatch_problems: set[
        tuple[
            int | str,
            str,
            tuple[str, ...],
            tuple[str, ...],
        ]
    ] = set()

    call_schemas: dict[
        tuple[int | str, str],
        set[tuple[str, ...]],
    ] = {}

    for problem in problems:
        interface = resolve_interface(
            problem,
            "python",
        )

        expected_names = [
            parameter.name
            for parameter
            in interface.parameters
        ]

        try:
            calls = _candidate_calls(
                problem.test
            )

        except Exception as exc:
            failures.append(
                (
                    f"{problem.question_id} "
                    f"{problem.task_id}: "
                    f"could not parse test: "
                    f"{type(exc).__name__}: {exc}"
                )
            )
            continue

        if not calls:
            failures.append(
                (
                    f"{problem.question_id} "
                    f"{problem.task_id}: "
                    "no candidate(...) calls found"
                )
            )
            continue

        for call in calls:
            total_calls += 1

            positional = list(
                call.args
            )

            keywords = [
                keyword
                for keyword
                in call.keywords
                if keyword.arg is not None
            ]

            has_star_kwargs = any(
                keyword.arg is None
                for keyword
                in call.keywords
            )

            if has_star_kwargs:
                failures.append(
                    (
                        f"{problem.question_id} "
                        f"{problem.task_id}: "
                        "candidate call uses **kwargs"
                    )
                )
                continue

            if (
                positional
                and keywords
            ):
                mixed_calls += 1

            elif positional:
                positional_calls += 1

            else:
                keyword_calls += 1

            supplied_count = (
                len(positional)
                + len(keywords)
            )

            arities[
                supplied_count
            ] += 1

            if supplied_count != len(
                expected_names
            ):
                failures.append(
                    (
                        f"{problem.question_id} "
                        f"{problem.task_id}: "
                        f"expected "
                        f"{len(expected_names)} args "
                        f"{expected_names}, "
                        f"got {supplied_count}"
                    )
                )
                continue

            positional_names = (
                expected_names[
                    :len(positional)
                ]
            )

            keyword_names = [
                keyword.arg
                for keyword
                in keywords
            ]

            problem_key = (
                problem.question_id,
                problem.task_id,
            )

            schema = tuple(
                keyword_names
            )

            call_schemas.setdefault(
                problem_key,
                set(),
            ).add(
                schema
            )

            supplied_names = (
                positional_names
                + keyword_names
            )

            if (
                len(set(supplied_names))
                != len(supplied_names)
            ):
                failures.append(
                    (
                        f"{problem.question_id} "
                        f"{problem.task_id}: "
                        "duplicate argument in "
                        "candidate call"
                    )
                )
                continue

            if set(
                supplied_names
            ) != set(
                expected_names
            ):
                mismatch_problems.add(
                    (
                        problem.question_id,
                        problem.task_id,
                        tuple(expected_names),
                        tuple(supplied_names),
                    )
                )

    inconsistent_schemas: list[
        tuple[
            int | str,
            str,
            set[tuple[str, ...]],
        ]
    ] = []

    for (
        question_id,
        task_id,
    ), schemas in sorted(
        call_schemas.items(),
        key=lambda item: int(
            item[0][0]
        ),
    ):
        if len(schemas) > 1:
            inconsistent_schemas.append(
                (
                    question_id,
                    task_id,
                    schemas,
                )
            )

    print()
    print("=" * 80)
    print("HF TEST CALL AUDIT")
    print("=" * 80)

    print(
        f"Problems: {len(problems)}"
    )

    print(
        f"candidate(...) calls: "
        f"{total_calls}"
    )

    print(
        f"positional-only calls: "
        f"{positional_calls}"
    )

    print(
        f"keyword-only calls: "
        f"{keyword_calls}"
    )

    print(
        f"mixed calls: "
        f"{mixed_calls}"
    )

    print()

    print("Call arities:")

    for arity, count in sorted(
        arities.items()
    ):
        print(
            f"  {arity}: {count}"
        )

    print()

    print(
        "Problems with argument-name "
        f"mismatch: {len(mismatch_problems)}"
    )

    for (
        question_id,
        task_id,
        interface_args,
        test_args,
    ) in sorted(
        mismatch_problems,
        key=lambda item: int(
            item[0]
        ),
    ):
        print(
            f"  {question_id} "
            f"{task_id}: "
            f"interface="
            f"{list(interface_args)} "
            f"tests="
            f"{list(test_args)}"
        )

    print()

    print(
        "Problems with inconsistent "
        f"keyword schemas: "
        f"{len(inconsistent_schemas)}"
    )

    for (
        question_id,
        task_id,
        schemas,
    ) in inconsistent_schemas:
        print(
            f"  {question_id} "
            f"{task_id}:"
        )

        for schema in sorted(
            schemas
        ):
            print(
                f"    {list(schema)}"
            )

    print()

    print(
        f"Structural failures: "
        f"{len(failures)}"
    )

    for failure in failures:
        print(
            f"  {failure}"
        )

    print()

    if (
        failures
        or inconsistent_schemas
    ):
        print("RESULT: FAIL")
        raise SystemExit(1)

    print("RESULT: PASS")


if __name__ == "__main__":
    main()