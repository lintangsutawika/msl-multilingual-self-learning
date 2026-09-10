from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict

from .dataset import load_test_split


def classify_entry_point(entry_point: str) -> str:
    if re.fullmatch(r"Solution\(\)\.[A-Za-z_]\w*", entry_point):
        return "solution_method"

    if re.fullmatch(r"[A-Za-z_]\w*", entry_point):
        return "function"

    return "other"


def detect_special_types(text: str) -> set[str]:
    types = set()

    for name in [
        "ListNode",
        "TreeNode",
        "Node",
        "NestedInteger",
    ]:
        if name in text:
            types.add(name)

    return types


def try_parse_input_output(value):
    if isinstance(value, list):
        return True, value

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return isinstance(parsed, list), parsed
        except Exception:
            return False, None

    return False, None


def main() -> None:
    problems = load_test_split()

    print(f"Total test problems: {len(problems)}")
    print()

    entry_point_types = Counter()
    difficulties = Counter()
    special_type_counts = Counter()
    bad_input_output = []
    empty_starter_code = []
    empty_entry_point = []
    starter_patterns = Counter()

    examples_by_special_type = defaultdict(list)

    for problem in problems:
        difficulties[problem.difficulty] += 1

        entry_type = classify_entry_point(problem.entry_point)
        entry_point_types[entry_type] += 1

        if not problem.starter_code.strip():
            empty_starter_code.append(problem.task_id)

        if not problem.entry_point.strip():
            empty_entry_point.append(problem.task_id)

        first_line = (
            problem.starter_code.strip().splitlines()[0]
            if problem.starter_code.strip()
            else "<EMPTY>"
        )
        starter_patterns[first_line] += 1

        combined_text = "\n".join(
            [
                problem.problem_description,
                problem.starter_code,
                problem.test,
            ]
        )

        special_types = detect_special_types(combined_text)

        for special_type in special_types:
            special_type_counts[special_type] += 1

            if len(examples_by_special_type[special_type]) < 10:
                examples_by_special_type[special_type].append(
                    problem.task_id
                )

        ok, parsed = try_parse_input_output(problem.input_output)

        if not ok:
            bad_input_output.append(problem.task_id)

    print("Difficulty distribution:")
    for name, count in difficulties.most_common():
        print(f"  {name}: {count}")

    print()
    print("Entry-point patterns:")
    for name, count in entry_point_types.most_common():
        print(f"  {name}: {count}")

    print()
    print("Starter-code first-line patterns:")
    for pattern, count in starter_patterns.most_common(20):
        print(f"  {count:3d}  {pattern}")

    print()
    print("Special types:")
    if special_type_counts:
        for name, count in special_type_counts.most_common():
            print(f"  {name}: {count}")
            for task_id in examples_by_special_type[name]:
                print(f"      {task_id}")
    else:
        print("  none")

    print()
    print(f"Empty starter_code: {len(empty_starter_code)}")
    for task_id in empty_starter_code[:20]:
        print(f"  {task_id}")

    print()
    print(f"Empty entry_point: {len(empty_entry_point)}")
    for task_id in empty_entry_point[:20]:
        print(f"  {task_id}")

    print()
    print(f"Unparseable input_output: {len(bad_input_output)}")
    for task_id in bad_input_output[:30]:
        print(f"  {task_id}")


if __name__ == "__main__":
    main()
    