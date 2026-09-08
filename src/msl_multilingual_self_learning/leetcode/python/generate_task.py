from __future__ import annotations

from pathlib import Path

from ..dataset import LeetCodeProblem


TASK_TOML_TEMPLATE = """schema_version = "1.0"

[task]
name = "leetcode/{task_id}-python"

[metadata]
language = "python"
difficulty = "{difficulty}"
source = "newfacade/LeetCodeDataset"
task_id = "{task_id}"

[agent]
timeout_sec = 300

[verifier]
timeout_sec = 60

[environment]
cpus = 1
memory_mb = 2048
"""


DOCKERFILE = """FROM python:3.12-slim

WORKDIR /workspace

CMD ["bash"]
"""


TEST_SH = """#!/usr/bin/env bash

set -uo pipefail

mkdir -p /logs/verifier

python /tests/test.py
STATUS=$?

case "$STATUS" in
    0)
        echo "1" > /logs/verifier/reward.txt
        echo "PASS" > /logs/verifier/status.txt
        ;;
    1)
        echo "0" > /logs/verifier/reward.txt
        echo "WRONG_ANSWER" > /logs/verifier/status.txt
        ;;
    *)
        echo "0" > /logs/verifier/reward.txt
        echo "RUNTIME_ERROR" > /logs/verifier/status.txt
        ;;
esac

exit 0
"""

SOLVE_SH = """#!/usr/bin/env bash
set -euo pipefail

cp /solution/solution.py /workspace/solution.py
"""

def render_instruction(problem: LeetCodeProblem) -> str:
    return f"""### Problem

{problem.problem_description}

### Required Interface

```python
{problem.starter_code}
```

### Instructions

Solve the problem in Python.

Preserve the required class and method interface.

Return only the final Python source code.
"""


def render_test_py(problem: LeetCodeProblem) -> str:
    return f"""import sys
import traceback

import random
import functools
import collections
import string
import math
import datetime

from typing import *
from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *

inf = float("inf")


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def list_node(values: list):
    if not values:
        return None

    head = ListNode(values[0])
    p = head

    for val in values[1:]:
        node = ListNode(val)
        p.next = node
        p = node

    return head


def is_same_list(p1, p2):
    if p1 is None and p2 is None:
        return True

    if not p1 or not p2:
        return False

    return (
        p1.val == p2.val
        and is_same_list(p1.next, p2.next)
    )


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def tree_node(values: list):
    if not values:
        return None

    root = TreeNode(values[0])
    i = 1
    queue = deque([root])

    while queue:
        node = queue.popleft()

        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)

        i += 1

        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)

        i += 1

    return root


def is_same_tree(p, q):
    if not p and not q:
        return True

    if not p or not q:
        return False

    if p.val != q.val:
        return False

    return (
        is_same_tree(p.left, q.left)
        and is_same_tree(p.right, q.right)
    )


SOLUTION_PATH = "/workspace/solution.py"


try:
    with open(SOLUTION_PATH, "r", encoding="utf-8") as f:
        solution_code = f.read()

    exec(solution_code, globals())

except Exception:
    print("LOAD_ERROR", file=sys.stderr)
    traceback.print_exc()
    sys.exit(2)


try:
    candidate = {problem.entry_point}

except Exception:
    print("ENTRY_POINT_ERROR", file=sys.stderr)
    traceback.print_exc()
    sys.exit(2)


{problem.test}


try:
    check(candidate)

except AssertionError:
    print("WRONG_ANSWER", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

except Exception:
    print("RUNTIME_ERROR", file=sys.stderr)
    traceback.print_exc()
    sys.exit(2)


print("PASS")
sys.exit(0)
"""


def generate_python_task(
    problem: LeetCodeProblem,
    output_root: Path,
) -> Path:
    task_dir = output_root / problem.task_id

    environment_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    solution_dir = task_dir / "solution"

    environment_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)
    solution_dir.mkdir(parents=True, exist_ok=True)

    (task_dir / "instruction.md").write_text(
        render_instruction(problem)
    )

    (task_dir / "task.toml").write_text(
        TASK_TOML_TEMPLATE.format(
            task_id=problem.task_id,
            difficulty=problem.difficulty.lower(),
        )
    )

    (environment_dir / "Dockerfile").write_text(
        DOCKERFILE
    )

    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(TEST_SH)
    test_sh_path.chmod(0o755)

    (tests_dir / "test.py").write_text(
        render_test_py(problem)
    )

    (solution_dir / "solution.py").write_text(
        problem.completion
    )

    solve_sh_path = solution_dir / "solve.sh"
    solve_sh_path.write_text(SOLVE_SH)
    solve_sh_path.chmod(0o755)

    return task_dir