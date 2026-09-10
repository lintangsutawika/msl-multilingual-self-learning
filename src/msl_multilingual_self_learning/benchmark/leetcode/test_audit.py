from __future__ import annotations

import ast
from collections import Counter

from .dataset import load_test_split


def main():
    problems = load_test_split()

    empty_tests = []
    syntax_failures = []
    missing_check = []

    assertion_counts = Counter()

    for problem in problems:
        test_code = problem.test

        if not test_code.strip():
            empty_tests.append(problem.task_id)
            continue

        try:
            tree = ast.parse(test_code)
        except SyntaxError as exc:
            syntax_failures.append(
                (
                    problem.task_id,
                    str(exc),
                )
            )
            continue

        found_check = False
        assert_count = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name == "check":
                    found_check = True

            if isinstance(node, ast.Assert):
                assert_count += 1

        if not found_check:
            missing_check.append(problem.task_id)

        assertion_counts[problem.task_id] = assert_count

    print(f"Total problems: {len(problems)}")

    print()
    print(f"Empty tests: {len(empty_tests)}")
    for task_id in empty_tests:
        print(f"  {task_id}")

    print()
    print(f"Syntax failures: {len(syntax_failures)}")
    for task_id, error in syntax_failures:
        print(f"  {task_id}")
        print(f"    {error}")

    print()
    print(f"Missing check(candidate): {len(missing_check)}")
    for task_id in missing_check:
        print(f"  {task_id}")

    counts = list(assertion_counts.values())

    if counts:
        print()
        print("Assertion statistics:")
        print(f"  min: {min(counts)}")
        print(f"  max: {max(counts)}")
        print(f"  total: {sum(counts)}")
        print(f"  mean: {sum(counts) / len(counts):.2f}")

        print()
        print("Problems with zero asserts:")
        for task_id, count in assertion_counts.items():
            if count == 0:
                print(f"  {task_id}")


if __name__ == "__main__":
    main()