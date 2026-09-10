from __future__ import annotations

import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

from .dataset import load_test_split


@dataclass
class TreeNode:
    val: int = 0
    left: TreeNode | None = None
    right: TreeNode | None = None


def tree_node(
    values: list[int | None],
) -> TreeNode | None:
    """
    Build a binary tree from the standard LeetCode level-order
    representation.

    Example:

        [1, 2, 3, None, 4]

    represents:

            1
           / \
          2   3
           \
            4
    """

    if not values:
        return None

    if values[0] is None:
        return None

    root = TreeNode(
        val=values[0]
    )

    queue: deque[TreeNode] = deque(
        [root]
    )

    index = 1

    while (
        queue
        and index < len(values)
    ):
        node = queue.popleft()

        if index < len(values):
            left_value = values[index]
            index += 1

            if left_value is not None:
                node.left = TreeNode(
                    val=left_value
                )

                queue.append(
                    node.left
                )

        if index < len(values):
            right_value = values[index]
            index += 1

            if right_value is not None:
                node.right = TreeNode(
                    val=right_value
                )

                queue.append(
                    node.right
                )

    return root


def _is_tree_node(
    value: Any,
) -> bool:
    return (
        hasattr(value, "val")
        and hasattr(value, "left")
        and hasattr(value, "right")
    )


def _jsonable(
    value: Any,
) -> Any:
    """
    Normalize a value into the representation used by the executor
    JSON boundary.

    This check is intended for values that actually cross that
    boundary: candidate input arguments.
    """

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        int,
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        return value

    if isinstance(
        value,
        str,
    ):
        return value

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            _jsonable(item)
            for item in value
        ]

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _jsonable(item)
            for key, item
            in value.items()
        }

    if _is_tree_node(
        value
    ):
        return {
            "val": _jsonable(
                value.val
            ),
            "left": _jsonable(
                value.left
            ),
            "right": _jsonable(
                value.right
            ),
        }

    raise TypeError(
        "Unsupported value type for "
        "JSON transport: "
        f"{type(value).__name__}"
    )


def _value_kind(
    value: Any,
) -> str:
    if value is None:
        return "None"

    if isinstance(
        value,
        bool,
    ):
        return "bool"

    if isinstance(
        value,
        int,
    ):
        return "int"

    if isinstance(
        value,
        float,
    ):
        if math.isnan(
            value
        ):
            return "float[nan]"

        if math.isinf(
            value
        ):
            if value > 0:
                return "float[+inf]"

            return "float[-inf]"

        return "float"

    if isinstance(
        value,
        str,
    ):
        return "str"

    if isinstance(
        value,
        list,
    ):
        if not value:
            return "list[empty]"

        child_kinds = {
            _value_kind(item)
            for item in value
        }

        if len(child_kinds) == 1:
            child_kind = next(
                iter(child_kinds)
            )

            return (
                f"list[{child_kind}]"
            )

        return (
            "list[mixed:"
            + ",".join(
                sorted(
                    child_kinds
                )
            )
            + "]"
        )

    if isinstance(
        value,
        tuple,
    ):
        if not value:
            return "tuple[empty]"

        child_kinds = {
            _value_kind(item)
            for item in value
        }

        if len(child_kinds) == 1:
            child_kind = next(
                iter(child_kinds)
            )

            return (
                f"tuple[{child_kind}]"
            )

        return (
            "tuple[mixed:"
            + ",".join(
                sorted(
                    child_kinds
                )
            )
            + "]"
        )

    if _is_tree_node(
        value
    ):
        return "TreeNode"

    return type(
        value
    ).__name__


class ResultProbe:
    """
    Dummy result returned by RecordingCandidate.

    Its equality operators record the value appearing on the other
    side of a canonical assertion such as:

        assert candidate(...) == expected

    Expected values are inspected for reporting only. They do not
    cross the executor JSON boundary.
    """

    def __init__(
        self,
        expected_values: list[Any],
    ) -> None:
        self.expected_values = (
            expected_values
        )

    def __eq__(
        self,
        other: Any,
    ) -> bool:
        self.expected_values.append(
            other
        )

        return True

    def __ne__(
        self,
        other: Any,
    ) -> bool:
        self.expected_values.append(
            other
        )

        return False

    def __bool__(
        self,
    ) -> bool:
        return True


class RecordingCandidate:
    def __init__(
        self,
    ) -> None:
        self.calls: list[
            tuple[
                tuple[Any, ...],
                dict[str, Any],
            ]
        ] = []

        self.expected_values: list[
            Any
        ] = []

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> ResultProbe:
        self.calls.append(
            (
                args,
                dict(kwargs),
            )
        )

        return ResultProbe(
            self.expected_values
        )


def _load_check(
    test_source: str,
):
    namespace: dict[str, Any] = {
        "TreeNode": TreeNode,
        "tree_node": tree_node,
        "inf": math.inf,
    }

    exec(
        test_source,
        namespace,
    )

    check = namespace.get(
        "check"
    )

    if not callable(
        check
    ):
        raise RuntimeError(
            "Test source does not "
            "define callable check(candidate)"
        )

    return check


def main() -> None:
    problems = load_test_split()

    total_calls = 0
    total_argument_values = 0
    total_expected_values = 0

    argument_kinds: Counter[
        str
    ] = Counter()

    expected_kinds: Counter[
        str
    ] = Counter()

    execution_failures: list[
        str
    ] = []

    argument_transport_failures: list[
        str
    ] = []

    positional_call_problems: set[
        tuple[
            int | str,
            str,
        ]
    ] = set()

    keyword_call_problems: set[
        tuple[
            int | str,
            str,
        ]
    ] = set()

    tree_argument_problems: set[
        tuple[
            int | str,
            str,
        ]
    ] = set()

    nonfinite_argument_problems: set[
        tuple[
            int | str,
            str,
        ]
    ] = set()

    nonfinite_expected_problems: set[
        tuple[
            int | str,
            str,
        ]
    ] = set()

    for problem in problems:
        problem_key = (
            problem.question_id,
            problem.task_id,
        )

        try:
            check = _load_check(
                problem.test
            )

            candidate = (
                RecordingCandidate()
            )

            check(
                candidate
            )

        except Exception as exc:
            execution_failures.append(
                (
                    f"{problem.question_id} "
                    f"{problem.task_id}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
            )
            continue

        for (
            args,
            kwargs,
        ) in candidate.calls:
            total_calls += 1

            if args:
                positional_call_problems.add(
                    problem_key
                )

            if kwargs:
                keyword_call_problems.add(
                    problem_key
                )

            values = (
                list(args)
                + list(
                    kwargs.values()
                )
            )

            for value in values:
                total_argument_values += 1

                kind = _value_kind(
                    value
                )

                argument_kinds[
                    kind
                ] += 1

                if kind == "TreeNode":
                    tree_argument_problems.add(
                        problem_key
                    )

                if (
                    isinstance(
                        value,
                        float,
                    )
                    and not math.isfinite(
                        value
                    )
                ):
                    nonfinite_argument_problems.add(
                        problem_key
                    )

                try:
                    normalized = _jsonable(
                        value
                    )

                    json.dumps(
                        normalized,
                        allow_nan=False,
                    )

                except Exception as exc:
                    argument_transport_failures.append(
                        (
                            f"{problem.question_id} "
                            f"{problem.task_id}: "
                            f"argument kind={kind}: "
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )
                    )

        for value in (
            candidate.expected_values
        ):
            total_expected_values += 1

            kind = _value_kind(
                value
            )

            expected_kinds[
                kind
            ] += 1

            if (
                isinstance(
                    value,
                    float,
                )
                and not math.isfinite(
                    value
                )
            ):
                nonfinite_expected_problems.add(
                    problem_key
                )

    print()
    print("=" * 80)
    print(
        "HF TEST VALUE / JSON "
        "TRANSPORT AUDIT"
    )
    print("=" * 80)

    print(
        f"Problems: "
        f"{len(problems)}"
    )

    print(
        f"Executed candidate calls: "
        f"{total_calls}"
    )

    print(
        f"Argument values inspected: "
        f"{total_argument_values}"
    )

    print(
        f"Expected values captured: "
        f"{total_expected_values}"
    )

    print()

    print(
        "Argument runtime kinds:"
    )

    for (
        kind,
        count,
    ) in sorted(
        argument_kinds.items()
    ):
        print(
            f"  {kind:<45} "
            f"{count}"
        )

    print()

    print(
        "Expected-value runtime kinds:"
    )

    for (
        kind,
        count,
    ) in sorted(
        expected_kinds.items()
    ):
        print(
            f"  {kind:<45} "
            f"{count}"
        )

    print()

    print(
        "Problems containing positional "
        f"candidate calls: "
        f"{len(positional_call_problems)}"
    )

    print(
        "Problems containing keyword "
        f"candidate calls: "
        f"{len(keyword_call_problems)}"
    )

    print(
        "Problems containing TreeNode "
        f"arguments: "
        f"{len(tree_argument_problems)}"
    )

    for (
        question_id,
        task_id,
    ) in sorted(
        tree_argument_problems,
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
        "Problems containing non-finite "
        f"argument values: "
        f"{len(nonfinite_argument_problems)}"
    )

    for (
        question_id,
        task_id,
    ) in sorted(
        nonfinite_argument_problems,
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
        "Problems containing non-finite "
        f"expected values: "
        f"{len(nonfinite_expected_problems)}"
    )

    for (
        question_id,
        task_id,
    ) in sorted(
        nonfinite_expected_problems,
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
        f"Test execution failures: "
        f"{len(execution_failures)}"
    )

    for failure in (
        execution_failures[:100]
    ):
        print(
            f"  {failure}"
        )

    if len(
        execution_failures
    ) > 100:
        print(
            f"  ... "
            f"{len(execution_failures) - 100} "
            "more"
        )

    print()

    print(
        "Argument JSON transport "
        f"failures: "
        f"{len(argument_transport_failures)}"
    )

    for failure in (
        argument_transport_failures[
            :100
        ]
    ):
        print(
            f"  {failure}"
        )

    if len(
        argument_transport_failures
    ) > 100:
        print(
            f"  ... "
            f"{len(argument_transport_failures) - 100} "
            "more"
        )

    print()

    if (
        execution_failures
        or argument_transport_failures
    ):
        print(
            "RESULT: FAIL"
        )

        raise SystemExit(
            1
        )

    if total_calls != 23234:
        print(
            "RESULT: FAIL"
        )

        print(
            "Expected 23234 canonical "
            "candidate calls, "
            f"captured {total_calls}."
        )

        raise SystemExit(
            1
        )

    print(
        "RESULT: PASS"
    )


if __name__ == "__main__":
    main()