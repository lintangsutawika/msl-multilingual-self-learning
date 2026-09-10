from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from .dataset import load_test_split
from .interfaces.resolve import resolve_interface
from .verifier.verifier import SolutionVerifier


@dataclass
class RecordingExecutor:
    last_args: tuple | None = None
    call_count: int = 0

    def call(
        self,
        *args,
    ):
        self.last_args = args
        self.call_count += 1

        return args

    def close(
        self,
    ) -> None:
        pass


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

        calls.append(
            node
        )

    return calls


def main() -> None:
    problems = load_test_split()

    total_calls = 0
    translated_calls = 0

    mismatched_name_calls = 0

    failures: list[str] = []

    mismatched_problems: set[
        tuple[
            int | str,
            str,
        ]
    ] = set()

    with TemporaryDirectory(
        prefix="leetcode-verifier-call-audit-"
    ) as temp_dir:
        workspace = Path(
            temp_dir
        )

        for problem in problems:
            interface = resolve_interface(
                problem,
                "python",
            )

            interface_names = [
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
                        f"failed to parse tests: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )
                )
                continue

            verifier = SolutionVerifier(
                workspace=workspace,
                interface=interface,
            )

            executor = RecordingExecutor()

            verifier.executor = executor

            for call_index, call in enumerate(
                calls
            ):
                total_calls += 1

                if call.args:
                    failures.append(
                        (
                            f"{problem.question_id} "
                            f"{problem.task_id}: "
                            "unexpected positional "
                            "arguments in canonical "
                            "test call"
                        )
                    )
                    continue

                if any(
                    keyword.arg is None
                    for keyword in call.keywords
                ):
                    failures.append(
                        (
                            f"{problem.question_id} "
                            f"{problem.task_id}: "
                            "canonical test call "
                            "uses **kwargs expansion"
                        )
                    )
                    continue

                keyword_names = [
                    keyword.arg
                    for keyword
                    in call.keywords
                ]

                if len(
                    keyword_names
                ) != len(
                    interface_names
                ):
                    failures.append(
                        (
                            f"{problem.question_id} "
                            f"{problem.task_id}: "
                            "argument-count mismatch: "
                            f"interface="
                            f"{interface_names}, "
                            f"tests="
                            f"{keyword_names}"
                        )
                    )
                    continue

                if (
                    keyword_names
                    != interface_names
                ):
                    mismatched_name_calls += 1

                    mismatched_problems.add(
                        (
                            problem.question_id,
                            problem.task_id,
                        )
                    )

                markers = [
                    (
                        problem.question_id,
                        call_index,
                        argument_index,
                        keyword_name,
                    )
                    for (
                        argument_index,
                        keyword_name,
                    )
                    in enumerate(
                        keyword_names
                    )
                ]

                kwargs = {
                    keyword_name: marker
                    for (
                        keyword_name,
                        marker,
                    )
                    in zip(
                        keyword_names,
                        markers,
                        strict=True,
                    )
                }

                try:
                    result = verifier.call(
                        **kwargs
                    )

                except Exception as exc:
                    failures.append(
                        (
                            f"{problem.question_id} "
                            f"{problem.task_id}: "
                            "verifier call failed: "
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )
                    )
                    continue

                expected = tuple(
                    markers
                )

                if executor.last_args != expected:
                    failures.append(
                        (
                            f"{problem.question_id} "
                            f"{problem.task_id}: "
                            "executor received wrong "
                            "argument order: "
                            f"expected={expected!r}, "
                            f"got="
                            f"{executor.last_args!r}"
                        )
                    )
                    continue

                if result != expected:
                    failures.append(
                        (
                            f"{problem.question_id} "
                            f"{problem.task_id}: "
                            "verifier returned wrong "
                            "result from executor: "
                            f"expected={expected!r}, "
                            f"got={result!r}"
                        )
                    )
                    continue

                translated_calls += 1

    print()
    print("=" * 80)
    print("VERIFIER CALL ADAPTER AUDIT")
    print("=" * 80)

    print(
        f"Problems: "
        f"{len(problems)}"
    )

    print(
        f"Canonical candidate calls: "
        f"{total_calls}"
    )

    print(
        f"Successfully translated calls: "
        f"{translated_calls}"
    )

    print(
        f"Calls with parameter-name "
        f"mismatch: "
        f"{mismatched_name_calls}"
    )

    print(
        f"Problems with parameter-name "
        f"mismatch: "
        f"{len(mismatched_problems)}"
    )

    for (
        question_id,
        task_id,
    ) in sorted(
        mismatched_problems,
        key=lambda item: int(
            item[0]
        ),
    ):
        print(
            f"  {question_id} "
            f"{task_id}"
        )

    print()

    print(
        f"Failures: "
        f"{len(failures)}"
    )

    for failure in failures[:100]:
        print(
            f"  {failure}"
        )

    if len(failures) > 100:
        print(
            f"  ... "
            f"{len(failures) - 100} "
            "more"
        )

    print()

    if failures:
        print(
            "RESULT: FAIL"
        )
        raise SystemExit(
            1
        )

    if (
        translated_calls
        != total_calls
    ):
        print(
            "RESULT: FAIL"
        )
        raise SystemExit(
            1
        )

    print(
        "RESULT: PASS"
    )


if __name__ == "__main__":
    main()