"""Dataset build: answers that exact `==` would wrongly reject get a fitting comparison."""
import unittest

from src.benchmarks.leetcode.hf_dataset.build_dataset import apply_comparison, comparison_for

SOURCE = "def check(candidate):\n    assert candidate(x = 1) == EXPECTED\n"


def passes(comparison, expected, answer):
    namespace = {}
    exec(apply_comparison(SOURCE.replace("EXPECTED", expected), comparison), namespace)
    try:
        namespace["check"](lambda x: answer)
    except AssertionError:
        return False
    return True


class ComparisonTests(unittest.TestCase):
    def test_comparison_follows_flags(self):
        self.assertEqual(comparison_for(["any_order", "decimal_answer"]), "float")
        self.assertEqual(comparison_for(["any_order", "has_images"]), "unordered")
        self.assertEqual(comparison_for(["has_images"]), "exact")

    def test_exact_source_is_unchanged(self):
        self.assertEqual(apply_comparison(SOURCE, "exact"), SOURCE)

    def test_unordered_frees_only_the_top_level_order(self):
        self.assertTrue(passes("unordered", "[[1, 2], [3]]", [[3], [1, 2]]))
        self.assertFalse(passes("unordered", "[[1, 2], [3]]", [[2, 1], [3]]))
        self.assertFalse(passes("unordered", "[1, 2]", [1, 2, 2]))

    def test_float_accepts_small_errors_only(self):
        self.assertTrue(passes("float", "2.4166666666666665", 2.41666699))
        self.assertTrue(passes("float", "[1.0, 2.5]", [1.000001, 2.5]))
        self.assertFalse(passes("float", "2.4166666666666665", 2.4167))
        self.assertFalse(passes("float", "[1.0, 2.5]", [1.0]))


if __name__ == "__main__":
    unittest.main()
