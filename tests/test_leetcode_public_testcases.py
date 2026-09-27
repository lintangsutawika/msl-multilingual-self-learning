import unittest

from src.benchmarks.leetcode.public_testcases import (
    build_public_test_cases,
    extract_example_outputs,
)


class PublicTestCaseTests(unittest.TestCase):
    def test_extracts_legacy_preformatted_outputs(self):
        content = """
        <p><strong class="example">Example 1:</strong></p>
        <pre><strong>Input:</strong> nums = [2,7], target = 9
        <strong>Output:</strong> [0,1]
        <strong>Explanation:</strong> The indices are 0 and 1.</pre>
        <p><strong class="example">Example 2:</strong></p>
        <pre><b>Input:</b> matrix = [[1,2],[3,4]]
        <b>Output:</b> [[1,2],
        [3,4]]</pre>
        """

        self.assertEqual(
            extract_example_outputs(content),
            ["[0,1]", "[[1,2],\n        [3,4]]"],
        )

    def test_extracts_current_wrapped_and_plain_outputs(self):
        content = """
        <div class="example-block">
          <p><strong>Input:</strong>
             <span class="example-io">n = 5</span></p>
          <p><strong>Output:</strong>
             <span class="example-io">[3,2,1]</span></p>
        </div>
        <div class="example-block">
          <p><strong>Input:</strong>
             <span class="example-io">n = 4</span></p>
          <p><strong>Output:</strong> [1,1]</p>
        </div>
        """

        self.assertEqual(
            extract_example_outputs(content),
            ["[3,2,1]", "[1,1]"],
        )

    def test_preserves_mismatches_for_auditing(self):
        cases, stats = build_public_test_cases(
            ["[1]", "[2]"],
            "<pre><strong>Output:</strong> 1</pre>",
        )

        self.assertEqual(
            cases,
            [
                {"input": "[1]", "output": "1"},
                {"input": "[2]", "output": None},
            ],
        )
        self.assertEqual(
            stats,
            {
                "input_count": 2,
                "output_count": 1,
                "paired_count": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
