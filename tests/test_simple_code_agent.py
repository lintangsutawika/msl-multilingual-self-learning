"""SimpleCodeAgent's prompt (from its config) and code extraction from a model reply."""
import pathlib
import tempfile
import types
import unittest

from msl_multilingual_self_learning.agents.simple_code_agent import DEFAULT_CONFIG, SimpleCodeAgent

TEMPLATES = pathlib.Path(__file__).resolve().parents[1] / "src" / "benchmarks" / "leetcode"
LANGUAGES = ["cpp", "go", "java", "javascript", "php", "python", "ruby", "rust", "typescript"]


def agent(config_file=None):
    return SimpleCodeAgent(logs_dir=pathlib.Path(tempfile.mkdtemp()), config_file=config_file)


def instruction(language):
    """The language's instruction.md template filled the way the task builder fills it."""
    text = (TEMPLATES / f"task-template-{language}" / "instruction.md").read_text()
    fills = {"language": language, "entrypoint": "int f(int x) {", "container": "Solution",
             "problem": "Return x.", "source_file": f"solution.{language}", "callable": "f"}
    for key, value in fills.items():
        text = text.replace("{" + key + "}", value)
    return text


class PromptTests(unittest.TestCase):
    def test_mini_swe_agent_submission_steps_are_dropped(self):
        for language in LANGUAGES:
            with self.subTest(language=language):
                system, user = agent()._messages(instruction(language))
                self.assertEqual(system["role"], "system")
                self.assertTrue(system["content"].startswith("You are a coding assistant."))
                self.assertTrue(user["content"].startswith(f"Language: {language}\n"))
                self.assertIn("Problem:\nReturn x.", user["content"])
                self.assertTrue(user["content"].endswith("Do not include unused\nimports."))
                for dropped in ("/workspace", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "issue the command"):
                    self.assertNotIn(dropped, user["content"])

    def test_instruction_without_the_marker_is_refused(self):
        with self.assertRaisesRegex(ValueError, "drop_task_from"):
            agent()._messages("Language: python\nProblem:\nReturn x.\n")

    def test_config_without_simple_code_agent_keys_is_refused(self):
        mini_swe_config = pathlib.Path(DEFAULT_CONFIG).with_name("leetcode.yaml")
        with self.assertRaisesRegex(ValueError, "not a SimpleCodeAgent config"):
            agent(str(mini_swe_config))


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
