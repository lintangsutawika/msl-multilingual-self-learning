"""Constraint audit: LeetCode's Constraints list becomes checks on canonical test inputs."""
import unittest

from src.benchmarks.leetcode.constraints import audit_problem, build_checks, constraint_items

PAGE = """
<p>You are given <code>queries</code> where <code>queries[i] = [l<sub>i</sub>, r<sub>i</sub>]</code>.</p>
<p><strong>Constraints:</strong></p>
<ul>
    <li><code>1 &lt;= n == nums.length &lt;= 10<sup>3</sup></code></li>
    <li><code>-10<sup>9</sup> &lt;= nums[i] &lt;= 10<sup>9</sup></code></li>
    <li><code>0 &lt;= l<sub>i</sub> &lt; r<sub>i</sub> &lt; n</code></li>
    <li><code>s</code> consists only of lowercase English letters.</li>
    <li>All <code>queries[i]</code> are unique.</li>
    <li>The input is generated such that something needs the solution to verify.</li>
</ul>
"""


def problem(tests):
    body = "".join(f"    assert candidate(nums = {n}, queries = {q}, s = {s!r}) == 0\n" for n, q, s in tests)
    return {
        "question_id": 1,
        "metadata": {"canonical_parameter_names": ["nums", "queries", "s"]},
        "interfaces": {"python": {"parameters": [{"name": "nums"}, {"name": "queries"}, {"name": "s"}]}},
        "canonical_tests": {"source": "def check(candidate):\n" + body},
    }


class ConstraintTests(unittest.TestCase):
    def test_items_keep_exponents(self):
        self.assertEqual(constraint_items(PAGE)[1], "-10**(9) <= nums[i] <= 10**(9)")

    def test_rules_become_checks_and_unknown_rules_are_reported(self):
        checks, uncheckable, unparsed = build_checks(1, PAGE, {"nums", "queries", "s"})
        self.assertEqual(len(checks), 5)
        self.assertEqual(uncheckable, [])
        self.assertEqual(unparsed, ["The input is generated such that something needs the solution to verify."])

    def test_violating_tests_are_listed_with_their_rules(self):
        report = audit_problem(problem([
            ([1, 2, 3], [[0, 2]], "ab"),              # valid
            ([1, 2, 3], [[2, 0]], "ab"),              # l < r broken
            ([5 * 10**9], [[0, 0]], "ab"),            # value range and l < r broken
            ([1, 2], [[0, 1], [0, 1]], "aB"),         # duplicate query, uppercase letter
        ]), PAGE)
        self.assertEqual(report["tests"], 4)
        violated = {t["test"]: set(t["violates"]) for t in report["invalid_tests"]}
        self.assertEqual(len(violated), 3)
        self.assertIn("0 <= li < ri < n", violated["assert candidate(nums=[1, 2, 3], queries=[[2, 0]], s='ab') == 0"])
        self.assertEqual(violated["assert candidate(nums=[1, 2], queries=[[0, 1], [0, 1]], s='aB') == 0"],
                         {"s consists only of lowercase English letters.", "All queries[i] are unique."})


if __name__ == "__main__":
    unittest.main()
