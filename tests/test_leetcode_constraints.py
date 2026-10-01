"""Constraint audit: LeetCode's Constraints list becomes checks on canonical test inputs."""
import unittest

from src.benchmarks.leetcode.hf_dataset.constraints import _auto_rule, audit_problem, build_checks, constraint_items

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

    def test_character_set_wordings(self):
        rules = [
            ("s consists of only English letters (both uppercase and lowercase), digits (0-9), plus '+', minus '-', or dot '.'.",
             "-1.5e+3", "1 2"),
            ("path consists of English letters, digits, period '.', slash '/' or '_'.", "/home/a_b/..", "/a b"),
            ("s consists of integers and operators ('+', '-', '*', '/') separated by some number of spaces.", " 3+5 / 2 ", "3+x"),
            ("s consists of parentheses only '()[]{}'.", "([{}])", "(a)"),
            ("s consists of English letters (lower-case and upper-case), ',' and '.'.", "Ab,c.", "a b"),
        ]
        for rule, good, bad in rules:
            check = _auto_rule(rule, {"s", "path"}, {})
            name = "path" if rule.startswith("path") else "s"
            self.assertTrue(check({name: good}), rule)
            self.assertFalse(check({name: bad}), rule)
        grid = _auto_rule("board consists of only lowercase and uppercase English letters.", {"board"}, {})
        self.assertTrue(grid({"board": [["a", "B"]]}))
        self.assertFalse(grid({"board": [["a", "1"]]}))

    def test_explicit_ranges_narrow_generic_words(self):
        digits = _auto_rule("s consists only of digits '0' to '4'.", {"s"}, {})
        self.assertTrue(digits({"s": "0431"}))
        self.assertFalse(digits({"s": "0451"}))
        letters = _auto_rule("s consists only of lowercase English letters 'a' to 'e'.", {"s"}, {})
        self.assertFalse(letters({"s": "abz"}))

    def test_list_items_with_attributes_are_read(self):
        page = '<p><strong>Constraints:</strong></p><ul><li data-stringify-border="0"><code>2 &lt;= n &lt;= 100</code></li></ul>'
        self.assertEqual(constraint_items(page), ["2 <= n <= 100"])

    def test_plain_text_caret_is_a_power(self):
        page = "<p><strong>Constraints:</strong></p><ul><li>1 &lt;= label &lt;= 10^6</li></ul>"
        self.assertEqual(constraint_items(page), ["1 <= label <= 10**6"])

    def test_rule_that_rejects_a_public_example_is_not_used(self):
        page = "<p><strong>Constraints:</strong></p><ul><li>1 &lt;= n &lt;= 3</li><li>n is even.</li></ul>"
        tests = "def check(candidate):\n    assert candidate(n = 1) == 1\n    assert candidate(n = 5) == 5\n"
        problem = {"question_id": 1, "metadata": {"canonical_parameter_names": ["n"]},
                   "canonical_tests": {"source": tests}}
        report = audit_problem(problem, page, examples=[{"n": 1}])
        self.assertEqual(report["check_errors"], {"n is even.": "rejects a public example"})
        self.assertEqual([t["violates"] for t in report["invalid_tests"]], [["1 <= n <= 3"]])

    def test_vague_character_sets_are_not_checked(self):
        self.assertIsNone(_auto_rule("s consists of English letters, digits, symbols and spaces.", {"s"}, {}))

    def test_neighbour_index_stays_inside_the_list(self):
        check = _auto_rule("triangle[i].length == triangle[i - 1].length + 1", {"triangle"}, {})
        self.assertTrue(check({"triangle": [[1], [2, 3], [4, 5, 6]]}))
        self.assertFalse(check({"triangle": [[1], [2, 3], [4, 5]]}))


if __name__ == "__main__":
    unittest.main()
