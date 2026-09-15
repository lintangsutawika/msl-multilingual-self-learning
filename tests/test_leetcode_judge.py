"""Exercise the submission/worker/judge boundary, including dynamic Python calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.benchmarks.leetcode.adapter import generate


class JudgeTests(unittest.TestCase):
    def run_judge(self, code, language="python", source=None):
        problem = {
            "question_id": 1, "task_id": "fixture", "difficulty": "Easy",
            "problem_description": "Test fixture", "metadata": {"canonical_parameter_names": ["hf_name"]},
            "interfaces": {"python": {"parameters": [{"name": "native_name", "type": "int"}],
                "return_type": "int", "callable": "solve", "container": "Solution",
                "raw_signature": "def solve(self, native_name: int) -> int:"}},
            "canonical_tests": {"source": source or "def check(candidate):\n    for i in range(4):\n        assert candidate(hf_name=i) == i * 2\n"},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = generate(problem, "python", root / "tasks")
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


if __name__ == "__main__":
    unittest.main()
