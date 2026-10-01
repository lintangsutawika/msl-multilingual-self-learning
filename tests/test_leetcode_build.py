"""Dataset build: answers that exact `==` would wrongly reject get a fitting comparison."""
import unittest

from src.benchmarks.leetcode.hf_dataset.clean import apply_comparison, apply_drop_categories, comparison_for, public_tests

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
        self.assertEqual(comparison_for(["unordered_groups"]), "unordered_groups")

    def test_exact_source_is_unchanged(self):
        self.assertEqual(apply_comparison(SOURCE, "exact"), SOURCE)

    def test_unordered_frees_only_the_top_level_order(self):
        self.assertTrue(passes("unordered", "[[1, 2], [3]]", [[3], [1, 2]]))
        self.assertFalse(passes("unordered", "[[1, 2], [3]]", [[2, 1], [3]]))
        self.assertFalse(passes("unordered", "[1, 2]", [1, 2, 2]))

    def test_unordered_groups_frees_group_and_item_order(self):
        self.assertTrue(passes("unordered_groups", '[["eat", "tea"], ["bat"]]', [["bat"], ["tea", "eat"]]))
        self.assertFalse(passes("unordered_groups", '[["eat", "tea"], ["bat"]]', [["bat", "tea"], ["eat"]]))

    def test_float_accepts_small_errors_only(self):
        self.assertTrue(passes("float", "2.4166666666666665", 2.41666699))
        self.assertTrue(passes("float", "[1.0, 2.5]", [1.000001, 2.5]))
        self.assertFalse(passes("float", "2.4166666666666665", 2.4167))
        self.assertFalse(passes("float", "[1.0, 2.5]", [1.0]))


class PublicTestTests(unittest.TestCase):
    def test_examples_are_taken_from_the_cleaned_tests_in_page_order(self):
        source = ("def check(candidate):\n    assert candidate(x = [1], k = 2) == 3\n"
                  "    assert candidate(x = [5], k = 0) == 5\n    assert candidate(x = [7], k = 1) == 8\n")
        examples = [{"x": [7], "k": 1}, {"x": [1], "k": 2}, {"x": [9], "k": 9}]
        public, missing = public_tests(source, ["x", "k"], examples)
        self.assertEqual(public, "def check(candidate):\n    assert candidate(x=[7], k=1) == 8\n"
                                 "    assert candidate(x=[1], k=2) == 3\n")
        self.assertEqual(missing, 1)
        self.assertEqual(public_tests(source, ["x", "k"], [{"x": [9], "k": 9}]), ("", 1))


class DropCategoryTests(unittest.TestCase):
    def report(self):
        return [
            {"question_id": 1, "drop": [], "flags": ["multiple_answers"]},
            {"question_id": 2, "drop": [], "flags": ["figure_reference", "has_images"]},
            {"question_id": 3, "drop": [], "flags": ["any_order"], "similar_to_test": ["two-sum"]},
            {"question_id": 4, "drop": ["premium"], "flags": []},
        ]

    def test_each_split_drops_its_own_categories(self):
        records = [{"question_id": q} for q in (1, 2, 3)]
        test, train = self.report(), self.report()
        self.assertEqual(apply_drop_categories("test", records, test), [{"question_id": 3}])
        self.assertEqual([e["drop"] for e in test], [["multiple_answers"], ["figure_reference"], [], ["premium"]])
        self.assertEqual(apply_drop_categories("train", records, train), records)
        self.assertEqual(train[2]["drop"], [])


if __name__ == "__main__":
    unittest.main()
