from __future__ import annotations

import ast
from collections import Counter, defaultdict

from .dataset import load_test_split


def parse_assignment_list(text: str) -> dict[str, object]:
    """
    Parse input strings like:

        n = 5, queries = [[2,4],[0,2]]

    into:

        {
            "n": 5,
            "queries": [[2,4],[0,2]]
        }

    without using eval().
    """

    fake_call = f"f({text})"

    tree = ast.parse(fake_call, mode="eval")

    if not isinstance(tree.body, ast.Call):
        raise ValueError("Input did not parse as function arguments.")

    result = {}

    # Positional args would be unexpected for this dataset.
    if tree.body.args:
        raise ValueError("Unexpected positional arguments.")

    for keyword in tree.body.keywords:
        if keyword.arg is None:
            raise ValueError("Unexpected **kwargs syntax.")

        result[keyword.arg] = ast.literal_eval(keyword.value)

    return result


def classify_value(value):
    if value is None:
        return "None"

    if isinstance(value, bool):
        return "bool"

    if isinstance(value, int):
        return "int"

    if isinstance(value, float):
        return "float"

    if isinstance(value, str):
        return "str"

    if isinstance(value, list):
        if not value:
            return "list[empty]"

        child_types = {
            classify_value(x)
            for x in value
        }

        if len(child_types) == 1:
            return f"list[{next(iter(child_types))}]"

        return "list[mixed:" + ",".join(
            sorted(child_types)
        ) + "]"

    if isinstance(value, dict):
        return "dict"

    if isinstance(value, tuple):
        return "tuple"

    return type(value).__name__


def main():
    problems = load_test_split()

    parse_failures = []
    output_parse_failures = []
    positional_failures = []

    input_value_types = Counter()
    output_value_types = Counter()

    input_names = Counter()

    examples_by_input_type = defaultdict(list)
    examples_by_output_type = defaultdict(list)

    test_case_count = 0

    for problem in problems:
        for case in problem.input_output:
            test_case_count += 1

            input_text = case["input"]
            output_text = case["output"]

            try:
                parsed_inputs = parse_assignment_list(input_text)
            except Exception as exc:
                parse_failures.append(
                    (
                        problem.task_id,
                        input_text,
                        repr(exc),
                    )
                )
                continue

            for name, value in parsed_inputs.items():
                input_names[name] += 1

                value_type = classify_value(value)
                input_value_types[value_type] += 1

                if len(
                    examples_by_input_type[value_type]
                ) < 5:
                    examples_by_input_type[
                        value_type
                    ].append(
                        (
                            problem.task_id,
                            name,
                            repr(value)[:150],
                        )
                    )

            try:
                output_value = ast.literal_eval(output_text)
            except Exception as exc:
                output_parse_failures.append(
                    (
                        problem.task_id,
                        output_text,
                        repr(exc),
                    )
                )
                continue

            output_type = classify_value(output_value)
            output_value_types[output_type] += 1

            if len(
                examples_by_output_type[output_type]
            ) < 5:
                examples_by_output_type[
                    output_type
                ].append(
                    (
                        problem.task_id,
                        repr(output_value)[:150],
                    )
                )

    print(f"Problems: {len(problems)}")
    print(f"Test cases: {test_case_count}")

    print()
    print("Input parse failures:")
    print(f"  {len(parse_failures)}")

    for task_id, text, error in parse_failures[:20]:
        print()
        print(f"  {task_id}")
        print(f"    input: {text}")
        print(f"    error: {error}")

    print()
    print("Output parse failures:")
    print(f"  {len(output_parse_failures)}")

    for task_id, text, error in output_parse_failures[:20]:
        print()
        print(f"  {task_id}")
        print(f"    output: {text}")
        print(f"    error: {error}")

    print()
    print("Input value types:")

    for type_name, count in input_value_types.most_common():
        print(f"  {count:5d}  {type_name}")

        for example in examples_by_input_type[type_name]:
            task_id, name, value = example
            print(
                f"         {task_id}: "
                f"{name} = {value}"
            )

    print()
    print("Output value types:")

    for type_name, count in output_value_types.most_common():
        print(f"  {count:5d}  {type_name}")

        for example in examples_by_output_type[type_name]:
            task_id, value = example
            print(
                f"         {task_id}: {value}"
            )


if __name__ == "__main__":
    main()