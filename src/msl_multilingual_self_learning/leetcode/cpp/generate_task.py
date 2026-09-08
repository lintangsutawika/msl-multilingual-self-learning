from __future__ import annotations

import ast
from pathlib import Path

from ..dataset import LeetCodeProblem

TASK_TOML_TEMPLATE = """schema_version = "1.0"

[task]
name = "leetcode/{task_id}-cpp"

[metadata]
language = "cpp"
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


DOCKERFILE = """FROM gcc:14

RUN apt-get update && \\
    apt-get install -y --no-install-recommends \\
        python3 \\
        python3-venv \\
        python3-pip \\
        tmux \\
        asciinema \\
        ca-certificates && \\
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["bash"]
"""


TEST_SH = """#!/usr/bin/env bash

set -uo pipefail

mkdir -p /logs/verifier

g++ \
    -std=c++17 \
    -O2 \
    -pipe \
    /tests/test.cpp \
    -o /workspace/test_runner

COMPILE_STATUS=$?

if [ "$COMPILE_STATUS" -ne 0 ]; then
    echo "0" > /logs/verifier/reward.txt
    echo "COMPILE_ERROR" > /logs/verifier/status.txt
    exit 0
fi

/workspace/test_runner
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

TYPE_MAP = {
    "int": "int",
    "str": "string",
    "bool": "bool",
    "float": "double",
    "List[int]": "vector<int>",
    "List[List[int]]": "vector<vector<int>>",
    "List[str]": "vector<string>",
    "List[List[str]]": "vector<vector<string>>",
    "List[float]": "vector<double>",
    "Optional[TreeNode]": "TreeNode*",
    "List[bool]": "vector<bool>",
}


def normalize_annotation(node: ast.expr | None) -> str:
    if node is None:
        raise ValueError("Missing type annotation.")

    return ast.unparse(node)


def extract_solution_method(
    starter_code: str,
) -> ast.FunctionDef:
    code = starter_code.rstrip()
    lines = code.splitlines()

    if not lines:
        raise ValueError("Empty starter code.")

    if (
        lines[-1].lstrip().startswith("def ")
        and lines[-1].rstrip().endswith(":")
    ):
        indent = len(lines[-1]) - len(lines[-1].lstrip())
        code += "\n" + " " * (indent + 4) + "pass"

    tree = ast.parse(code)

    for node in tree.body:
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "Solution"
        ):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    return item

    raise ValueError(
        "Could not find Solution method."
    )


def cpp_type(annotation: str) -> str:
    try:
        return TYPE_MAP[annotation]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported C++ type annotation: {annotation}"
        ) from exc


def render_cpp_interface(
    problem: LeetCodeProblem,
) -> str:
    method = extract_solution_method(
        problem.starter_code
    )

    return_type = cpp_type(
        normalize_annotation(method.returns)
    )

    parameters = []

    for arg in method.args.args:
        if arg.arg == "self":
            continue

        annotation = normalize_annotation(
            arg.annotation
        )
        type_name = cpp_type(annotation)

        parameters.append(
            f"{type_name} {arg.arg}"
        )

    parameter_text = ", ".join(parameters)

    interface = f"""class Solution {{
public:
    {return_type} {method.name}({parameter_text}) {{
        // your implementation
    }}
}};"""

    uses_tree = any(
        normalize_annotation(arg.annotation)
        == "Optional[TreeNode]"
        for arg in method.args.args
        if arg.arg != "self"
    )

    if uses_tree:
        return f"""// TreeNode is provided by the judge.
struct TreeNode {{
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode(int x)
        : val(x), left(nullptr), right(nullptr) {{}}
}};

{interface}"""

    return interface


def render_instruction(
    problem: LeetCodeProblem,
) -> str:
    interface = render_cpp_interface(problem)

    return f"""### Problem

{problem.problem_description}

### Required Interface

```cpp
{interface}
```

### Instructions

Solve the problem in C++.

Preserve the required class and method interface.

Return only the final C++ source code.
"""


def cpp_literal(
    node: ast.expr,
    annotation: str,
) -> str:
    # Python verifier uses:
    #     inf = float("inf")
    #
    # Some dataset expected values are `inf` or `-inf`.
    if isinstance(node, ast.Name) and node.id == "inf":
        if annotation == "int":
            return "numeric_limits<int>::max()"

        if annotation == "float":
            return "numeric_limits<double>::infinity()"

        raise ValueError(
            f"inf is unsupported for annotation {annotation}"
        )

    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Name)
        and node.operand.id == "inf"
    ):
        if annotation == "int":
            return "numeric_limits<int>::min()"

        if annotation == "float":
            return "-numeric_limits<double>::infinity()"

        raise ValueError(
            f"-inf is unsupported for annotation {annotation}"
        )

    if annotation == "Optional[TreeNode]":
        return cpp_tree_literal(node)
        
    value = ast.literal_eval(node)

    if annotation == "int":
        return str(value)

    if annotation == "bool":
        return "true" if value else "false"

    if annotation == "float":
        return repr(float(value))

    if annotation == "str":
        return repr(value).replace("'", '"')

    if annotation == "List[int]":
        values = ", ".join(str(x) for x in value)
        return f"vector<int>{{{values}}}"

    if annotation == "List[float]":
        values = ", ".join(
            repr(float(x))
            for x in value
        )
        return f"vector<double>{{{values}}}"

    if annotation == "List[str]":
        values = ", ".join(
            repr(x).replace("'", '"')
            for x in value
        )
        return f"vector<string>{{{values}}}"

    if annotation == "List[List[int]]":
        rows = []

        for row in value:
            row_values = ", ".join(
                str(x)
                for x in row
            )
            rows.append(
                f"vector<int>{{{row_values}}}"
            )

        return (
            "vector<vector<int>>{"
            + ", ".join(rows)
            + "}"
        )

    if annotation == "List[List[str]]":
        rows = []

        for row in value:
            row_values = ", ".join(
                repr(x).replace("'", '"')
                for x in row
            )
            rows.append(
                f"vector<string>{{{row_values}}}"
            )

        return (
            "vector<vector<string>>{"
            + ", ".join(rows)
            + "}"
        )
    if annotation == "List[bool]":
        values = ", ".join(
            "true" if x else "false"
            for x in value
        )
        return f"vector<bool>{{{values}}}"

    raise ValueError(
        f"Unsupported literal type: {annotation}"
    )

def cpp_tree_literal(node: ast.expr) -> str:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "tree_node"
        and len(node.args) == 1
    ):
        raise ValueError(
            "Expected tree_node([...])"
        )

    values = ast.literal_eval(node.args[0])

    items = []

    for value in values:
        if value is None:
            items.append("nullopt")
        else:
            items.append(str(value))

    return (
        "buildTree(vector<optional<int>>{"
        + ", ".join(items)
        + "})"
    )

def render_cpp_test(
    problem: LeetCodeProblem,
) -> str:
    method = extract_solution_method(
        problem.starter_code
    )

    parameter_types = {}

    for arg in method.args.args:
        if arg.arg == "self":
            continue

        parameter_types[arg.arg] = normalize_annotation(
            arg.annotation
        )

    return_annotation = normalize_annotation(
        method.returns
    )

    tree = ast.parse(problem.test)

    test_lines = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue

        comparison = node.test

        if not (
            isinstance(comparison, ast.Compare)
            and len(comparison.ops) == 1
            and isinstance(comparison.ops[0], ast.Eq)
            and len(comparison.comparators) == 1
        ):
            raise ValueError(
                f"Unsupported assertion in {problem.task_id}"
            )

        call = comparison.left
        expected = comparison.comparators[0]

        if not isinstance(call, ast.Call):
            raise ValueError(
                f"Expected candidate call in {problem.task_id}"
            )

        keyword_values = {}

        for keyword in call.keywords:
            if keyword.arg is None:
                raise ValueError(
                    "Unsupported **kwargs."
                )

            keyword_values[keyword.arg] = keyword.value

        arguments = []

        for arg in method.args.args:
            if arg.arg == "self":
                continue

            if arg.arg not in keyword_values:
                raise ValueError(
                    f"Missing argument {arg.arg} "
                    f"in {problem.task_id}"
                )

            annotation = parameter_types[arg.arg]

            arguments.append(
                cpp_literal(
                    keyword_values[arg.arg],
                    annotation,
                )
            )

        expected_cpp = cpp_literal(
            expected,
            return_annotation,
        )

        call_cpp = (
            f"solution.{method.name}("
            + ", ".join(arguments)
            + ")"
        )

        test_number = len(test_lines) + 1

        test_lines.append(
            f"""    if (!({call_cpp} == {expected_cpp})) {{
        cerr << "WRONG_ANSWER test {test_number}" << endl;
        return 1;
    }}"""
        )

    tests = "\n\n".join(test_lines)

    tree_support = ""

    if "Optional[TreeNode]" in parameter_types.values():
        tree_support = r"""
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode(int x)
        : val(x), left(nullptr), right(nullptr) {}
};

TreeNode* buildTree(
    const vector<optional<int>>& values
) {
    if (
        values.empty()
        || !values[0].has_value()
    ) {
        return nullptr;
    }

    TreeNode* root = new TreeNode(
        values[0].value()
    );

    queue<TreeNode*> q;
    q.push(root);

    size_t i = 1;

    while (!q.empty() && i < values.size()) {
        TreeNode* node = q.front();
        q.pop();

        if (
            i < values.size()
            && values[i].has_value()
        ) {
            node->left = new TreeNode(
                values[i].value()
            );
            q.push(node->left);
        }

        ++i;

        if (
            i < values.size()
            && values[i].has_value()
        ) {
            node->right = new TreeNode(
                values[i].value()
            );
            q.push(node->right);
        }

        ++i;
    }

    return root;
}
"""

    return f"""#include <bits/stdc++.h>
using namespace std;

{tree_support}

#include "/workspace/solution.cpp"


int main() {{
    Solution solution;

{tests}

    cout << "PASS" << endl;
    return 0;
}}
"""

def generate_cpp_task(
    problem: LeetCodeProblem,
    output_root: Path,
) -> Path:
    task_dir = output_root / problem.task_id

    environment_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"

    environment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    tests_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    (tests_dir / "test.cpp").write_text(
        render_cpp_test(problem)
    )

    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(TEST_SH)
    test_sh_path.chmod(0o755)

    return task_dir