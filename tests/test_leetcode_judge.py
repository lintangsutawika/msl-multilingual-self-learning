"""Exercise the submission/worker/judge boundary, including dynamic Python calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import importlib.util

from src.benchmarks.leetcode.adapter import generate, generate_all

VERIFIER = Path(__file__).parents[1] / "src/benchmarks/leetcode/task-template-python/tests/test.py"


def load_verifier():
    """The verifier every task runs (identical in all nine templates)."""
    spec = importlib.util.spec_from_file_location("leetcode_verifier", VERIFIER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class JudgeTests(unittest.TestCase):
    def run_judge(self, code, language="python", source=None):
        row = {
            "question_id": 1, "task_id": "fixture", "difficulty": "Easy", "language": "python",
            "problem_description": "Test fixture", "metadata": {"canonical_parameter_names": ["hf_name"]},
            "interface": {"parameters": [{"name": "native_name", "type": "int"}],
                "return_type": "int", "callable": "solve", "container": "Solution",
                "raw_signature": "def solve(self, native_name: int) -> int:"},
            "canonical_tests": {"source": source or "def check(candidate):\n    for i in range(4):\n        assert candidate(hf_name=i) == i * 2\n"},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tasks").mkdir()
            task = generate(row, root / "tasks")
            workspace = root / "workspace"
            workspace.mkdir()
            logs = root / "logs"
            (workspace / "solution.json").write_text(json.dumps({"language": language, "code": code}))
            result = subprocess.run([sys.executable, str(task / "tests/test.py")], env={**os.environ,
                "LEETCODE_WORKSPACE": str(workspace), "LEETCODE_LOGS": str(logs),
                "LEETCODE_ADAPTERS": str(task / "environment/files/adapters")}, capture_output=True, text=True, timeout=20)
            return result, (logs / "status.txt").read_text().strip(), (logs / "reward.txt").read_text().strip()

    def test_dynamic_calls_and_native_argument_names(self):
        result, status, reward = self.run_judge("class Solution:\n    def solve(self, native_name):\n        return native_name * 2\n")
        self.assertEqual((result.returncode, status, reward), (0, "PASS", "1"), result.stderr)

    def test_solution_instance_persists(self):
        code = "class Solution:\n    def __init__(self): self.n = 0\n    def solve(self, native_name):\n        self.n += 1\n        return self.n\n"
        result, status, _ = self.run_judge(code, source="def check(candidate):\n    assert candidate(0) == 1\n    assert candidate(0) == 2\n")
        self.assertEqual(status, "PASS", result.stderr)

    def test_wrong_answer_has_zero_reward(self):
        _, status, reward = self.run_judge("class Solution:\n    def solve(self, native_name): return -1\n")
        self.assertEqual((status, reward), ("WRONG_ANSWER", "0"))

    def test_wrong_submission_language_is_rejected(self):
        _, status, reward = self.run_judge("unused", language="go")
        self.assertEqual((status, reward), ("RUNTIME_ERROR", "0"))

    def test_worker_crash_is_not_wrong_answer(self):
        _, status, reward = self.run_judge("class Solution:\n    def solve(self, native_name): raise ValueError('crash')\n")
        self.assertEqual((status, reward), ("RUNTIME_ERROR", "0"))


class NormalizeTests(unittest.TestCase):
    """Submission normalization: any reasonable file structure is accepted."""

    def setUp(self):
        self.judge = load_verifier()

    def test_go_package_renamed_and_user_main_kept_out_of_the_way(self):
        code, notes = self.judge.normalize("go", "package solution\n\nfunc f() {}\nfunc main() {}\n", {})
        self.assertTrue(code.startswith("package main\n"))
        self.assertIn("func leetcodeUserMain()", code)
        self.assertEqual(notes, ["go: renamed package to main", "go: renamed user main"])

    def test_go_missing_package_clause_is_added(self):
        code, _ = self.judge.normalize("go", "func f() {}\n", {})
        self.assertTrue(code.startswith("package main\n"))

    def test_go_imports_follow_compiler_errors(self):
        source = 'package main\n\nimport (\n\t"fmt"\n\th "container/heap"\n)\n'
        errors = '"fmt" imported and not used\n"container/heap" imported as h and not used\nundefined: sort\nundefined: x\n'
        notes = []
        fixed, changed = self.judge._fix_go_imports(source, errors, notes)
        self.assertTrue(changed)
        self.assertEqual(fixed, 'package main\nimport "sort"\n\nimport (\n)\n')

    def test_php_open_tag_added_once(self):
        self.assertTrue(self.judge.normalize("php", "class Solution {}", {})[0].startswith("<?php\n"))
        self.assertEqual(self.judge.normalize("php", "<?php\nclass Solution {}", {}), ("<?php\nclass Solution {}", []))

    def test_java_gets_leetcode_imports_without_package(self):
        code, notes = self.judge.normalize("java", "package a.b;\nclass Solution {}\n", {})
        self.assertTrue(code.startswith(self.judge.JAVA_IMPORTS))
        self.assertNotIn("package a.b", code)
        self.assertEqual(notes, ["java: removed package declaration"])

    def test_cpp_free_function_is_wrapped(self):
        config = {"callable": "solve", "container": "Solution"}
        code, notes = self.judge.normalize("cpp", "int solve(int x) { return x; }\n", config)
        self.assertIn("class Solution", code)
        self.assertEqual(notes, ["cpp: wrapped free function in Solution"])
        self.assertEqual(self.judge.normalize("cpp", "struct Solution { int solve(int x); };", config)[1], [])

    def test_rust_runner_follows_solution_structure(self):
        config = {"callable": "solve", "container": "Solution"}
        runner = "struct Solution;\n\ninclude!(\"solution.rs\");\nlet result = Solution::solve(arg0);\n"
        notes = []
        adapted = self.judge.adapt_runner("rust", runner, "pub struct Solution;\npub fn solve(x: i32) -> i32 { x }\n", config, notes)
        self.assertNotIn("struct Solution;", adapted)
        self.assertIn("let result = solve(arg0);", adapted)
        self.assertEqual(len(notes), 2)
        self.assertEqual(self.judge.adapt_runner("rust", runner, "impl Solution { fn solve() {} }", config, []), runner)


class OverflowFilterTests(unittest.TestCase):
    """Test cases that no declared LeetCode type can hold are dropped for every language."""

    def problem(self, source):
        interface = lambda t, r: {"parameters": [{"name": "x", "type": t}], "return_type": r}
        return {
            "metadata": {"canonical_parameter_names": ["x"]},
            "interfaces": {
                "python": interface("List[int]", "int"),
                "go": interface("[]int", "int64"),
                "rust": interface("Vec<i32>", "i64"),
                "java": interface("int[]", "long"),
            },
            "canonical_tests": {"source": source},
        }

    def test_out_of_range_input_is_dropped(self):
        from src.benchmarks.leetcode.hf_dataset.constraints import filter_canonical_tests
        source = (
            "def check(candidate):\n"
            "    assert candidate(x = [1, 2]) == 3\n"
            "    assert candidate(x = [3000000000]) == 3000000000\n"
            "    assert candidate([-2147483648]) == -2147483648\n"
        )
        filtered, dropped, kept = filter_canonical_tests(self.problem(source))
        self.assertEqual(filtered, "def check(candidate):\n    assert candidate(x = [1, 2]) == 3\n    assert candidate([-2147483648]) == -2147483648\n")
        self.assertEqual(dropped, [{"line": 3, "end_line": 3, "unrepresentable_in": ["rust", "java"]}])
        self.assertEqual(kept, 2)

    def test_large_result_fits_a_64_bit_return_type(self):
        from src.benchmarks.leetcode.hf_dataset.constraints import filter_canonical_tests
        source = "def check(candidate):\n    assert candidate(x = [2000000000, 2000000000]) == 4000000000\n"
        self.assertEqual(filter_canonical_tests(self.problem(source)), (source, [], 1))

    def test_infinite_expected_output_is_dropped_for_every_language(self):
        from src.benchmarks.leetcode.hf_dataset.constraints import filter_canonical_tests
        source = "def check(candidate):\n    assert candidate(x = [1]) == 1\n    assert candidate(x = [2]) == -inf\n"
        filtered, dropped, _ = filter_canonical_tests(self.problem(source))
        self.assertEqual(filtered, "def check(candidate):\n    assert candidate(x = [1]) == 1\n")
        self.assertEqual(dropped[0]["unrepresentable_in"], ["python", "go", "rust", "java"])

    def test_audited_invalid_tests_are_dropped(self):
        from src.benchmarks.leetcode.hf_dataset.constraints import filter_canonical_tests
        source = "def check(candidate):\n    assert candidate(x = [1]) == 1\n    assert candidate(x = [0]) == 0\n"
        invalid = {"assert candidate(x=[0]) == 0": ["1 <= x[i] <= 10"]}
        filtered, dropped, kept = filter_canonical_tests(self.problem(source), invalid)
        self.assertEqual(filtered, "def check(candidate):\n    assert candidate(x = [1]) == 1\n")
        self.assertEqual(dropped[0]["violates"], ["1 <= x[i] <= 10"])
        self.assertEqual(kept, 1)

    def test_unreadable_test_is_dropped(self):
        from src.benchmarks.leetcode.hf_dataset.constraints import filter_canonical_tests
        source = "def check(candidate):\n    assert candidate(x = [1]) == 1\n    assert candidate(x = [1, ..., 9]) == 9\n"
        filtered, dropped, kept = filter_canonical_tests(self.problem(source))
        self.assertEqual(filtered, "def check(candidate):\n    assert candidate(x = [1]) == 1\n")
        self.assertEqual((dropped[0]["unreadable"], kept), (True, 1))

    def test_every_test_dropped_leaves_none(self):
        from src.benchmarks.leetcode.hf_dataset.constraints import filter_canonical_tests
        self.assertEqual(filter_canonical_tests(self.problem("def check(candidate):\n    assert candidate(x = [2**40]) == 1\n"))[2], 0)


class GenerateAllTests(unittest.TestCase):
    """Tasks use the dataset's (already cleaned) tests as they are."""

    def rows(self):
        source = ("def check(candidate):\n"
                  + "".join(f"    assert candidate(x = [{i}]) == {i}\n" for i in range(11)))
        base = {"question_id": 7, "task_id": "fixture", "difficulty": "Easy", "problem_description": "Fixture",
                "metadata": {"canonical_parameter_names": ["x"]}, "canonical_tests": {"source": source}}
        interface = lambda t, sig: {"parameters": [{"name": "x", "type": t}], "return_type": "int",
                                    "callable": "f", "container": "Solution", "raw_signature": sig}
        return [
            {**base, "language": "python", "interface": interface("List[int]", "def f(self, x: List[int]) -> int:")},
            {**base, "language": "java", "interface": interface("int[]", "public int f(int[] x) {")},
        ]

    def test_tests_are_used_as_they_are_and_languages_filter_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "tasks"
            _, count = generate_all(self.rows(), out, languages=("python",))
            self.assertEqual(count, 1)
            self.assertEqual((out / "7-python/tests/canonical_test.py").read_text(),
                             self.rows()[0]["canonical_tests"]["source"])


if __name__ == "__main__":
    unittest.main()
