from __future__ import annotations

import shutil
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .executors.cpp import CppExecutor
from .executors.go import GoExecutor
from .executors.java import JavaExecutor
from .executors.python import PythonExecutor
from .interfaces.models import Parameter, SolutionInterface


@dataclass
class TreeNode:
    val: int
    left: TreeNode | None = None
    right: TreeNode | None = None


@dataclass(frozen=True)
class RuntimeCase:
    name: str
    language: str
    parameter_type: str
    return_type: str
    input_value: Any
    expected_value: Any
    code: str


EXECUTORS = {
    "python": PythonExecutor,
    "cpp": CppExecutor,
    "go": GoExecutor,
    "java": JavaExecutor,
}


def _interface(
    language: str,
    parameter_type: str,
    return_type: str,
) -> SolutionInterface:
    container = (
        None
        if language == "go"
        else "Solution"
    )

    return SolutionInterface(
        question_id=-1,
        language=language,
        container=container,
        callable_name="probe",
        parameters=(
            Parameter(
                name="value",
                type=parameter_type,
            ),
        ),
        return_type=return_type,
        source="runtime-audit",
        raw_signature=None,
    )


def _python_cases() -> list[RuntimeCase]:
    return [
        RuntimeCase(
            name="python-int",
            language="python",
            parameter_type="int",
            return_type="int",
            input_value=123456789,
            expected_value=123456789,
            code="""
class Solution:
    def probe(self, value):
        return value
""",
        ),
        RuntimeCase(
            name="python-str",
            language="python",
            parameter_type="str",
            return_type="str",
            input_value="hello",
            expected_value="hello",
            code="""
class Solution:
    def probe(self, value):
        return value
""",
        ),
        RuntimeCase(
            name="python-bool",
            language="python",
            parameter_type="bool",
            return_type="bool",
            input_value=True,
            expected_value=True,
            code="""
class Solution:
    def probe(self, value):
        return value
""",
        ),
        RuntimeCase(
            name="python-float",
            language="python",
            parameter_type="float",
            return_type="float",
            input_value=1.25,
            expected_value=1.25,
            code="""
class Solution:
    def probe(self, value):
        return value
""",
        ),
        RuntimeCase(
            name="python-list-int",
            language="python",
            parameter_type="List[int]",
            return_type="List[int]",
            input_value=[1, -2, 3],
            expected_value=[1, -2, 3],
            code="""
class Solution:
    def probe(self, value):
        return value
""",
        ),
        RuntimeCase(
            name="python-list-str",
            language="python",
            parameter_type="List[str]",
            return_type="List[str]",
            input_value=["a", "bc"],
            expected_value=["a", "bc"],
            code="""
class Solution:
    def probe(self, value):
        return value
""",
        ),
        RuntimeCase(
            name="python-list-bool",
            language="python",
            parameter_type="List[bool]",
            return_type="List[bool]",
            input_value=[True, False, True],
            expected_value=[True, False, True],
            code="""
class Solution:
    def probe(self, value):
        return value
""",
        ),
        RuntimeCase(
            name="python-list-float",
            language="python",
            parameter_type="List[float]",
            return_type="float",
            input_value=[1.25, 2.5],
            expected_value=3.75,
            code="""
class Solution:
    def probe(self, value):
        return sum(value)
""",
        ),
        RuntimeCase(
            name="python-matrix-int",
            language="python",
            parameter_type="List[List[int]]",
            return_type="List[List[int]]",
            input_value=[[1, 2], [3, 4]],
            expected_value=[[1, 2], [3, 4]],
            code="""
class Solution:
    def probe(self, value):
        return value
""",
        ),
        RuntimeCase(
            name="python-matrix-str",
            language="python",
            parameter_type="List[List[str]]",
            return_type="str",
            input_value=[["a", "b"], ["c"]],
            expected_value="a",
            code="""
class Solution:
    def probe(self, value):
        return value[0][0]
""",
        ),
        RuntimeCase(
            name="python-tree",
            language="python",
            parameter_type="Optional[TreeNode]",
            return_type="int",
            input_value=TreeNode(
                10,
                TreeNode(5),
                TreeNode(20),
            ),
            expected_value=10,
            code="""
class Solution:
    def probe(self, value):
        return value.val
""",
        ),
    ]


def _cpp_cases() -> list[RuntimeCase]:
    def solution(
        parameter_type: str,
        return_type: str,
        body: str = "return value;",
    ) -> str:
        return f"""
#include <bits/stdc++.h>
using namespace std;

class Solution {{
public:
    {return_type} probe({parameter_type} value) {{
        {body}
    }}
}};
"""

    return [
        RuntimeCase(
            "cpp-int",
            "cpp",
            "int",
            "int",
            123456789,
            123456789,
            solution("int", "int"),
        ),
        RuntimeCase(
            "cpp-long-long",
            "cpp",
            "long long",
            "long long",
            5_000_000_000,
            5_000_000_000,
            solution(
                "long long",
                "long long",
            ),
        ),
        RuntimeCase(
            "cpp-string",
            "cpp",
            "string",
            "string",
            "hello",
            "hello",
            solution(
                "string",
                "string",
            ),
        ),
        RuntimeCase(
            "cpp-std-string",
            "cpp",
            "std::string",
            "string",
            "hello",
            "hello",
            solution(
                "std::string",
                "string",
            ),
        ),
        RuntimeCase(
            "cpp-bool",
            "cpp",
            "bool",
            "bool",
            True,
            True,
            solution(
                "bool",
                "bool",
            ),
        ),
        RuntimeCase(
            "cpp-double",
            "cpp",
            "double",
            "double",
            1.25,
            1.25,
            solution(
                "double",
                "double",
            ),
        ),
        RuntimeCase(
            "cpp-vector-int-ref",
            "cpp",
            "vector<int>&",
            "vector<int>",
            [1, -2, 3],
            [1, -2, 3],
            solution(
                "vector<int>&",
                "vector<int>",
            ),
        ),
        RuntimeCase(
            "cpp-vector-int-ref-spacing",
            "cpp",
            "vector<int> &",
            "vector<int>",
            [1, 2],
            [1, 2],
            solution(
                "vector<int> &",
                "vector<int>",
            ),
        ),
        RuntimeCase(
            "cpp-vector-long-long-ref",
            "cpp",
            "vector<long long>&",
            "vector<long long>",
            [5_000_000_000, 6_000_000_000],
            [5_000_000_000, 6_000_000_000],
            solution(
                "vector<long long>&",
                "vector<long long>",
            ),
        ),
        RuntimeCase(
            "cpp-vector-double-ref",
            "cpp",
            "vector<double>&",
            "double",
            [1.25, 2.5],
            3.75,
            solution(
                "vector<double>&",
                "double",
                "return value[0] + value[1];",
            ),
        ),
        RuntimeCase(
            "cpp-vector-string-ref",
            "cpp",
            "vector<string>&",
            "vector<string>",
            ["a", "bc"],
            ["a", "bc"],
            solution(
                "vector<string>&",
                "vector<string>",
            ),
        ),
        RuntimeCase(
            "cpp-vector-bool",
            "cpp",
            "vector<bool>",
            "vector<bool>",
            [True, False, True],
            [True, False, True],
            solution(
                "vector<bool>",
                "vector<bool>",
            ),
        ),
        RuntimeCase(
            "cpp-matrix-int-ref",
            "cpp",
            "vector<vector<int>>&",
            "vector<vector<int>>",
            [[1, 2], [3, 4]],
            [[1, 2], [3, 4]],
            solution(
                "vector<vector<int>>&",
                "vector<vector<int>>",
            ),
        ),
        RuntimeCase(
            "cpp-matrix-string-ref",
            "cpp",
            "vector<vector<string>>&",
            "string",
            [["a", "b"], ["c"]],
            "a",
            solution(
                "vector<vector<string>>&",
                "string",
                "return value[0][0];",
            ),
        ),
        RuntimeCase(
            "cpp-char-return",
            "cpp",
            "int",
            "char",
            1,
            "q",
            solution(
                "int",
                "char",
                "return 'q';",
            ),
        ),
        RuntimeCase(
            name="cpp-tree",
            language="cpp",
            parameter_type="TreeNode*",
            return_type="int",
            input_value=TreeNode(
                10,
                TreeNode(5),
                TreeNode(20),
            ),
            expected_value=10,
            code="""
#include <bits/stdc++.h>
using namespace std;

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;

    TreeNode(int x)
        : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    int probe(TreeNode* value) {
        return value->val;
    }
};
""",
        ),
    ]


def _go_cases() -> list[RuntimeCase]:
    def solution(
        parameter_type: str,
        return_type: str,
        body: str = "return value",
    ) -> str:
        return f"""
package main

func probe(value {parameter_type}) {return_type} {{
    {body}
}}
"""

    return [
        RuntimeCase(
            "go-int",
            "go",
            "int",
            "int",
            123456789,
            123456789,
            solution("int", "int"),
        ),
        RuntimeCase(
            "go-int64",
            "go",
            "int64",
            "int64",
            5_000_000_000,
            5_000_000_000,
            solution("int64", "int64"),
        ),
        RuntimeCase(
            "go-string",
            "go",
            "string",
            "string",
            "hello",
            "hello",
            solution("string", "string"),
        ),
        RuntimeCase(
            "go-bool",
            "go",
            "bool",
            "bool",
            True,
            True,
            solution("bool", "bool"),
        ),
        RuntimeCase(
            "go-float64",
            "go",
            "float64",
            "float64",
            1.25,
            1.25,
            solution(
                "float64",
                "float64",
            ),
        ),
        RuntimeCase(
            "go-slice-int",
            "go",
            "[]int",
            "[]int",
            [1, -2, 3],
            [1, -2, 3],
            solution("[]int", "[]int"),
        ),
        RuntimeCase(
            "go-slice-int64",
            "go",
            "[]int64",
            "[]int64",
            [5_000_000_000, 6_000_000_000],
            [5_000_000_000, 6_000_000_000],
            solution(
                "[]int64",
                "[]int64",
            ),
        ),
        RuntimeCase(
            "go-slice-string",
            "go",
            "[]string",
            "[]string",
            ["a", "bc"],
            ["a", "bc"],
            solution(
                "[]string",
                "[]string",
            ),
        ),
        RuntimeCase(
            "go-slice-bool",
            "go",
            "[]bool",
            "[]bool",
            [True, False],
            [True, False],
            solution(
                "[]bool",
                "[]bool",
            ),
        ),
        RuntimeCase(
            "go-slice-float64",
            "go",
            "[]float64",
            "float64",
            [1.25, 2.5],
            3.75,
            solution(
                "[]float64",
                "float64",
                "return value[0] + value[1]",
            ),
        ),
        RuntimeCase(
            "go-matrix-int",
            "go",
            "[][]int",
            "[][]int",
            [[1, 2], [3, 4]],
            [[1, 2], [3, 4]],
            solution(
                "[][]int",
                "[][]int",
            ),
        ),
        RuntimeCase(
            "go-matrix-string",
            "go",
            "[][]string",
            "string",
            [["a", "b"], ["c"]],
            "a",
            solution(
                "[][]string",
                "string",
                "return value[0][0]",
            ),
        ),
        RuntimeCase(
            "go-byte-return",
            "go",
            "int",
            "byte",
            1,
            "q",
            solution(
                "int",
                "byte",
                "return 'q'",
            ),
        ),
        RuntimeCase(
            name="go-tree",
            language="go",
            parameter_type="*TreeNode",
            return_type="int",
            input_value=TreeNode(
                10,
                TreeNode(5),
                TreeNode(20),
            ),
            expected_value=10,
            code="""
package main

type TreeNode struct {
    Val int
    Left *TreeNode
    Right *TreeNode
}

func probe(value *TreeNode) int {
    return value.Val
}
""",
        ),
    ]


def _java_cases() -> list[RuntimeCase]:
    def solution(
        parameter_type: str,
        return_type: str,
        body: str = "return value;",
    ) -> str:
        return f"""
import java.util.*;

class Solution {{
    public {return_type} probe(
        {parameter_type} value
    ) {{
        {body}
    }}
}}
"""

    return [
        RuntimeCase(
            "java-int",
            "java",
            "int",
            "int",
            123456789,
            123456789,
            solution("int", "int"),
        ),
        RuntimeCase(
            "java-long",
            "java",
            "long",
            "long",
            5_000_000_000,
            5_000_000_000,
            solution("long", "long"),
        ),
        RuntimeCase(
            "java-string",
            "java",
            "String",
            "String",
            "hello",
            "hello",
            solution(
                "String",
                "String",
            ),
        ),
        RuntimeCase(
            "java-boolean",
            "java",
            "boolean",
            "boolean",
            True,
            True,
            solution(
                "boolean",
                "boolean",
            ),
        ),
        RuntimeCase(
            "java-double",
            "java",
            "double",
            "double",
            1.25,
            1.25,
            solution(
                "double",
                "double",
            ),
        ),
        RuntimeCase(
            "java-int-array",
            "java",
            "int[]",
            "int[]",
            [1, -2, 3],
            [1, -2, 3],
            solution(
                "int[]",
                "int[]",
            ),
        ),
        RuntimeCase(
            "java-long-array",
            "java",
            "long[]",
            "long[]",
            [5_000_000_000, 6_000_000_000],
            [5_000_000_000, 6_000_000_000],
            solution(
                "long[]",
                "long[]",
            ),
        ),
        RuntimeCase(
            "java-string-array",
            "java",
            "String[]",
            "String",
            ["a", "bc"],
            "a",
            solution(
                "String[]",
                "String",
                "return value[0];",
            ),
        ),
        RuntimeCase(
            "java-boolean-array",
            "java",
            "boolean[]",
            "boolean[]",
            [True, False],
            [True, False],
            solution(
                "boolean[]",
                "boolean[]",
            ),
        ),
        RuntimeCase(
            "java-double-array",
            "java",
            "double[]",
            "double",
            [1.25, 2.5],
            3.75,
            solution(
                "double[]",
                "double",
                "return value[0] + value[1];",
            ),
        ),
        RuntimeCase(
            "java-int-matrix",
            "java",
            "int[][]",
            "int[][]",
            [[1, 2], [3, 4]],
            [[1, 2], [3, 4]],
            solution(
                "int[][]",
                "int[][]",
            ),
        ),
        RuntimeCase(
            "java-string-matrix",
            "java",
            "String[][]",
            "String",
            [["a", "b"], ["c"]],
            "a",
            solution(
                "String[][]",
                "String",
                "return value[0][0];",
            ),
        ),
        RuntimeCase(
            "java-list-integer",
            "java",
            "List<Integer>",
            "List<Integer>",
            [1, -2, 3],
            [1, -2, 3],
            solution(
                "List<Integer>",
                "List<Integer>",
            ),
        ),
        RuntimeCase(
            "java-list-string",
            "java",
            "List<String>",
            "List<String>",
            ["a", "bc"],
            ["a", "bc"],
            solution(
                "List<String>",
                "List<String>",
            ),
        ),
        RuntimeCase(
            "java-list-list-integer",
            "java",
            "List<List<Integer>>",
            "List<Integer>",
            [[1, 2], [3]],
            [1, 2],
            solution(
                "List<List<Integer>>",
                "List<Integer>",
                "return value.get(0);",
            ),
        ),
        RuntimeCase(
            "java-list-list-string",
            "java",
            "List<List<String>>",
            "List<String>",
            [["a", "b"], ["c"]],
            ["a", "b"],
            solution(
                "List<List<String>>",
                "List<String>",
                "return value.get(0);",
            ),
        ),
        RuntimeCase(
            "java-char-return",
            "java",
            "int",
            "char",
            1,
            "q",
            solution(
                "int",
                "char",
                "return 'q';",
            ),
        ),
        RuntimeCase(
            name="java-tree",
            language="java",
            parameter_type="TreeNode",
            return_type="int",
            input_value=TreeNode(
                10,
                TreeNode(5),
                TreeNode(20),
            ),
            expected_value=10,
            code="""
class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
}

class Solution {
    public int probe(TreeNode value) {
        return value.val;
    }
}
""",
        ),
    ]


def _all_cases() -> list[RuntimeCase]:
    return (
        _python_cases()
        + _cpp_cases()
        + _go_cases()
        + _java_cases()
    )


def _runtime_available(
    language: str,
) -> tuple[bool, str | None]:
    if language == "python":
        return True, None

    if language == "cpp":
        if shutil.which("g++") is None:
            return False, "g++ not found"
        return True, None

    if language == "go":
        if shutil.which("go") is None:
            return False, "go not found"
        return True, None

    if language == "java":
        if shutil.which("javac") is None:
            return False, "javac not found"

        if shutil.which("java") is None:
            return False, "java not found"

        try:
            from .executors.java import _find_gson_jar

            _find_gson_jar()
        except Exception as exc:
            return (
                False,
                f"Gson unavailable: {exc}",
            )

        return True, None

    return False, "unknown language"


def _equal(
    actual: Any,
    expected: Any,
) -> bool:
    if (
        isinstance(expected, float)
        and isinstance(actual, (int, float))
    ):
        return abs(
            float(actual) - expected
        ) <= 1e-9

    return actual == expected


def _run_case(
    case: RuntimeCase,
) -> tuple[str, str]:
    available, reason = (
        _runtime_available(
            case.language
        )
    )

    if not available:
        return (
            "SKIP",
            reason or "runtime unavailable",
        )

    interface = _interface(
        language=case.language,
        parameter_type=case.parameter_type,
        return_type=case.return_type,
    )

    executor_type = EXECUTORS[
        case.language
    ]

    with tempfile.TemporaryDirectory(
        prefix=(
            f"leetcode-runtime-audit-"
            f"{case.language}-"
        )
    ) as temporary_directory:
        workspace = Path(
            temporary_directory
        )

        executor = executor_type(
            workspace=workspace,
            interface=interface,
        )

        try:
            executor.prepare(
                case.code.strip() + "\n"
            )

            actual = executor.call(
                case.input_value
            )

            if not _equal(
                actual,
                case.expected_value,
            ):
                return (
                    "FAIL",
                    (
                        f"expected="
                        f"{case.expected_value!r}, "
                        f"actual={actual!r}, "
                        f"actual_type="
                        f"{type(actual).__name__}"
                    ),
                )

            return (
                "PASS",
                (
                    f"value={actual!r}"
                ),
            )

        except Exception:
            return (
                "ERROR",
                traceback.format_exc(),
            )

        finally:
            try:
                executor.close()
            except Exception:
                pass


def main() -> None:
    cases = _all_cases()

    counts = {
        "PASS": 0,
        "FAIL": 0,
        "ERROR": 0,
        "SKIP": 0,
    }

    failures: list[
        tuple[RuntimeCase, str, str]
    ] = []

    print()
    print("=" * 80)
    print("EXECUTOR RUNTIME TYPE AUDIT")
    print("=" * 80)
    print()

    current_language = None

    for case in cases:
        if case.language != current_language:
            current_language = (
                case.language
            )

            print()
            print(
                f"[{current_language}]"
            )

        status, detail = _run_case(
            case
        )

        counts[status] += 1

        print(
            f"  {status:5s} "
            f"{case.name}"
        )

        if status != "PASS":
            failures.append(
                (
                    case,
                    status,
                    detail,
                )
            )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for status in (
        "PASS",
        "FAIL",
        "ERROR",
        "SKIP",
    ):
        print(
            f"{status:5s}: "
            f"{counts[status]}"
        )

    if failures:
        print()
        print("=" * 80)
        print("DETAILS")
        print("=" * 80)

        for (
            case,
            status,
            detail,
        ) in failures:
            print()
            print(
                f"{status}: "
                f"{case.name}"
            )
            print(
                f"parameter="
                f"{case.parameter_type}"
            )
            print(
                f"return="
                f"{case.return_type}"
            )
            print(detail)

    if (
        counts["FAIL"] == 0
        and counts["ERROR"] == 0
        and counts["SKIP"] == 0
    ):
        print()
        print("RESULT: PASS")
        return

    print()
    print("RESULT: FAIL")
    raise SystemExit(1)


if __name__ == "__main__":
    main()