from __future__ import annotations

import ast
import json
from pathlib import Path

from ..dataset import LeetCodeProblem

TASK_TOML_TEMPLATE = '''schema_version = "1.0"

[task]
name = "leetcode/{task_id}-javascript"

[metadata]
language = "javascript"
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
'''

DOCKERFILE = """FROM node:22-bookworm

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

TEST_SH = '''#!/usr/bin/env bash
set -uo pipefail
mkdir -p /logs/verifier

node /tests/test.js
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
'''


def normalize_annotation(node: ast.expr | None) -> str:
    if node is None:
        raise ValueError("Missing type annotation.")
    return ast.unparse(node)


def extract_solution_method(starter_code: str) -> ast.FunctionDef:
    code = starter_code.rstrip()
    lines = code.splitlines()
    if not lines:
        raise ValueError("Empty starter code.")

    if lines[-1].lstrip().startswith("def ") and lines[-1].rstrip().endswith(":"):
        indent = len(lines[-1]) - len(lines[-1].lstrip())
        code += "\n" + " " * (indent + 4) + "pass"

    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    return item
    raise ValueError("Could not find Solution method.")


def render_js_interface(problem: LeetCodeProblem) -> str:
    method = extract_solution_method(problem.starter_code)
    params = [arg.arg for arg in method.args.args if arg.arg != "self"]
    parameter_text = ", ".join(params)

    uses_tree = any(
        normalize_annotation(arg.annotation) == "Optional[TreeNode]"
        for arg in method.args.args
        if arg.arg != "self"
    )

    tree_definition = ""
    if uses_tree:
        tree_definition = '''class TreeNode {
    constructor(val, left = null, right = null) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

'''

    return f'''{tree_definition}class Solution {{
    {method.name}({parameter_text}) {{
        // your implementation
    }}
}}'''


def render_instruction(problem: LeetCodeProblem) -> str:
    interface = render_js_interface(problem)
    return f'''### Problem

{problem.problem_description}

### Required Interface

```javascript
{interface}
```

### Instructions

Solve the problem in JavaScript.

Preserve the required class and method interface.

Return only the final JavaScript source code.
'''


def js_value(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(js_value(x) for x in value) + "]"
    raise ValueError(f"Unsupported JavaScript literal value: {value!r}")


def js_tree_literal(node: ast.expr) -> str:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "tree_node"
        and len(node.args) == 1
    ):
        raise ValueError("Expected tree_node([...])")
    values = ast.literal_eval(node.args[0])
    return f"buildTree({js_value(values)})"


def js_literal(node: ast.expr, annotation: str) -> str:
    if annotation == "Optional[TreeNode]":
        return js_tree_literal(node)

    if isinstance(node, ast.Name):
        if node.id == "inf":
            return "Infinity"
        raise ValueError(f"Unsupported JavaScript name: {node.id}")

    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Name)
        and node.operand.id == "inf"
    ):
        return "-Infinity"

    return js_value(ast.literal_eval(node))


def render_js_test(problem: LeetCodeProblem) -> str:
    method = extract_solution_method(problem.starter_code)

    parameter_types = {
        arg.arg: normalize_annotation(arg.annotation)
        for arg in method.args.args
        if arg.arg != "self"
    }
    return_annotation = normalize_annotation(method.returns)

    tree = ast.parse(problem.test)
    test_lines: list[str] = []

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
            raise ValueError(f"Unsupported assertion in {problem.task_id}")

        call = comparison.left
        expected = comparison.comparators[0]
        if not isinstance(call, ast.Call):
            raise ValueError(f"Expected candidate call in {problem.task_id}")

        keyword_values = {}
        for keyword in call.keywords:
            if keyword.arg is None:
                raise ValueError("Unsupported **kwargs.")
            keyword_values[keyword.arg] = keyword.value

        arguments = []
        for arg in method.args.args:
            if arg.arg == "self":
                continue
            if arg.arg not in keyword_values:
                raise ValueError(f"Missing argument {arg.arg} in {problem.task_id}")
            arguments.append(js_literal(keyword_values[arg.arg], parameter_types[arg.arg]))

        expected_js = js_literal(expected, return_annotation)
        call_js = f"solution.{method.name}(" + ", ".join(arguments) + ")"
        n = len(test_lines) + 1

        test_lines.append(f'''    if (!deepEqual({call_js}, {expected_js})) {{
        console.error("WRONG_ANSWER test {n}");
        process.exit(1);
    }}''')

    tests = "\n\n".join(test_lines)

    tree_support = ""
    if "Optional[TreeNode]" in parameter_types.values():
        tree_support = r'''
function buildTree(values) {
    if (values.length === 0 || values[0] === null) {
        return null;
    }

    const root = new TreeNode(values[0]);
    const queue = [root];
    let qIndex = 0;
    let i = 1;

    while (qIndex < queue.length && i < values.length) {
        const node = queue[qIndex++];

        if (i < values.length && values[i] !== null) {
            node.left = new TreeNode(values[i]);
            queue.push(node.left);
        }
        i += 1;

        if (i < values.length && values[i] !== null) {
            node.right = new TreeNode(values[i]);
            queue.push(node.right);
        }
        i += 1;
    }

    return root;
}
'''

    return f'''const fs = require("fs");
const vm = require("vm");

const source = fs.readFileSync("/workspace/solution.js", "utf8");
vm.runInThisContext(source, {{ filename: "/workspace/solution.js" }});

function deepEqual(a, b) {{
    if (typeof a === "number" && typeof b === "number") {{
        if (Number.isNaN(a) && Number.isNaN(b)) {{
            return true;
        }}
        return Object.is(a, b);
    }}

    if (Array.isArray(a) && Array.isArray(b)) {{
        if (a.length !== b.length) {{
            return false;
        }}
        for (let i = 0; i < a.length; i += 1) {{
            if (!deepEqual(a[i], b[i])) {{
                return false;
            }}
        }}
        return true;
    }}

    return a === b;
}}

{tree_support}

const solution = new Solution();

{tests}

console.log("PASS");
process.exit(0);
'''


def generate_js_task(problem: LeetCodeProblem, output_root: Path) -> Path:
    task_dir = output_root / problem.task_id
    environment_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"

    environment_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    (task_dir / "instruction.md").write_text(render_instruction(problem))
    (task_dir / "task.toml").write_text(
        TASK_TOML_TEMPLATE.format(
            task_id=problem.task_id,
            difficulty=problem.difficulty.lower(),
        )
    )
    (environment_dir / "Dockerfile").write_text(DOCKERFILE)
    (tests_dir / "test.js").write_text(render_js_test(problem))

    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(TEST_SH)
    test_sh_path.chmod(0o755)

    return task_dir
