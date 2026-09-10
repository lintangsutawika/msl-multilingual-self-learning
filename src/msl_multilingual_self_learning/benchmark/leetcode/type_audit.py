from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict

from .dataset import load_test_split


def normalize_annotation(node: ast.expr | None) -> str:
    if node is None:
        return "<missing>"

    try:
        return ast.unparse(node)
    except Exception:
        return "<unparseable>"


def extract_solution_method(starter_code: str):
    """
    Parse the Python starter code and return the first method
    defined inside class Solution.

    The dataset's starter snippets usually have an empty function body,
    so we add `pass` before parsing.
    """

    code = starter_code.rstrip()

    lines = code.splitlines()

    if not lines:
        return None

    # If the last meaningful line is a function definition ending in ':',
    # give it a valid body.
    if lines[-1].lstrip().startswith("def ") and lines[-1].rstrip().endswith(":"):
        indent = len(lines[-1]) - len(lines[-1].lstrip())
        code += "\n" + " " * (indent + 4) + "pass"

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            for item in node.body:
                if isinstance(
                    item,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    return item

    return None


def detect_explicit_special_types(starter_code: str) -> set[str]:
    """
    Look only for explicit type names rather than substrings
    in the natural-language problem description.
    """

    special_types = set()

    for name in [
        "ListNode",
        "TreeNode",
        "Node",
        "NestedInteger",
    ]:
        if re.search(rf"\b{re.escape(name)}\b", starter_code):
            special_types.add(name)

    return special_types


def classify_python_value(value):
    """
    Recursively describe the shape of an actual parsed Python value.
    """

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

        element_types = {
            classify_python_value(element)
            for element in value
        }

        if len(element_types) == 1:
            element_type = next(iter(element_types))
            return f"list[{element_type}]"

        return "list[mixed:" + ",".join(
            sorted(element_types)
        ) + "]"

    if isinstance(value, tuple):
        return "tuple"

    if isinstance(value, dict):
        return "dict"

    return type(value).__name__


def main():
    problems = load_test_split()

    argument_annotations = Counter()
    return_annotations = Counter()
    method_names = Counter()

    signature_patterns = Counter()
    special_types = Counter()

    parse_failures = []
    missing_solution_method = []

    examples_by_signature = defaultdict(list)
    examples_by_special_type = defaultdict(list)

    for problem in problems:
        method = extract_solution_method(problem.starter_code)

        if method is None:
            missing_solution_method.append(problem.task_id)
            continue

        method_names[method.name] += 1

        args = []

        for arg in method.args.args:
            # Skip self.
            if arg.arg == "self":
                continue

            annotation = normalize_annotation(arg.annotation)
            argument_annotations[annotation] += 1
            args.append(annotation)

        return_annotation = normalize_annotation(method.returns)
        return_annotations[return_annotation] += 1

        signature = (
            "("
            + ", ".join(args)
            + ") -> "
            + return_annotation
        )

        signature_patterns[signature] += 1

        if len(examples_by_signature[signature]) < 5:
            examples_by_signature[signature].append(
                problem.task_id
            )

        detected = detect_explicit_special_types(
            problem.starter_code
        )

        for special_type in detected:
            special_types[special_type] += 1

            if len(
                examples_by_special_type[special_type]
            ) < 10:
                examples_by_special_type[special_type].append(
                    problem.task_id
                )

    print(f"Total problems: {len(problems)}")

    print()
    print("Method extraction failures:")
    print(f"  {len(missing_solution_method)}")
    for task_id in missing_solution_method:
        print(f"    {task_id}")

    print()
    print("Argument annotations:")
    for annotation, count in argument_annotations.most_common():
        print(f"  {count:4d}  {annotation}")

    print()
    print("Return annotations:")
    for annotation, count in return_annotations.most_common():
        print(f"  {count:4d}  {annotation}")

    print()
    print("Explicit special types:")
    if special_types:
        for type_name, count in special_types.most_common():
            print(f"  {type_name}: {count}")
            for task_id in examples_by_special_type[type_name]:
                print(f"      {task_id}")
    else:
        print("  none")

    print()
    print(
        f"Unique signature patterns: "
        f"{len(signature_patterns)}"
    )

    print()
    print("Most common signatures:")

    for signature, count in signature_patterns.most_common(30):
        print()
        print(f"  {count:3d} × {signature}")

        for task_id in examples_by_signature[signature]:
            print(f"        {task_id}")


if __name__ == "__main__":
    main()