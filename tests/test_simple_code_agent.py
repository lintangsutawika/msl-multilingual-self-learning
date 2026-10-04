"""SimpleCodeAgent's code extraction from a model reply."""
import types
import unittest

from msl_multilingual_self_learning.agents.simple_code_agent import SimpleCodeAgent


def extract(reply):
    return SimpleCodeAgent._strip_markdown_fences(types.SimpleNamespace(), reply)


class ExtractCodeTests(unittest.TestCase):
    def test_reasoning_and_fences_are_removed(self):
        self.assertEqual(extract("<think>plan</think>\n```cpp\nint f();\n```"), "int f();")

    def test_mini_swe_agent_marker_is_removed(self):
        marker = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        self.assertEqual(extract(f"class S {{\n}};\n\n{marker}\n"), "class S {\n};")
        self.assertEqual(extract(f"```php\n<?php\nclass S {{}}\n{marker}\n```"), "<?php\nclass S {}")

    def test_marker_only_fence_does_not_replace_the_code(self):
        reply = "```go\nfunc f() int { return 1 }\n```\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"
        self.assertEqual(extract(reply), "func f() int { return 1 }")

    def test_marker_text_inside_code_is_kept(self):
        code = "def f():\n    print('echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT')"
        self.assertEqual(extract(f"```python\n{code}\n```"), code)


if __name__ == "__main__":
    unittest.main()
