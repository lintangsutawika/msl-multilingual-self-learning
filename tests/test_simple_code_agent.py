"""SimpleCodeAgent's prompt (from its config) and code extraction from a model reply."""
import asyncio
import json
import os
import pathlib
import tempfile
import types
import unittest
from unittest import mock

from msl_multilingual_self_learning.agents import simple_code_agent
from msl_multilingual_self_learning.agents.simple_code_agent import (
    DEFAULT_CONFIG, EmptySolutionError, ResponseTruncatedError, SimpleCodeAgent, bare_model_name, extract_code)

REPO = pathlib.Path(__file__).resolve().parents[1]
TEMPLATES = REPO / "src" / "benchmarks" / "leetcode"
LANGUAGES = ["cpp", "go", "java", "javascript", "php", "python", "ruby", "rust", "typescript"]
QWEN_9B = REPO / "configs" / "sampling" / "Qwen" / "Qwen3.5-9B.yaml"


def agent(config_file=None, sampling_file=QWEN_9B):
    return SimpleCodeAgent(logs_dir=pathlib.Path(tempfile.mkdtemp()), config_file=config_file,
                           sampling_file=str(sampling_file) if sampling_file else None)


def sampling(text):
    path = pathlib.Path(tempfile.mkdtemp()) / "sampling.yaml"
    path.write_text(text)
    return path


class SamplingTests(unittest.TestCase):
    def test_qwen_sampling_file_is_sent_like_mini_swe_agent(self):
        a = agent()
        self.assertEqual(a._client_kwargs, {"timeout": 1200.0})
        self.assertEqual(a._request_kwargs, {
            "max_tokens": 32768, "temperature": 0.6, "top_p": 0.95,
            "extra_body": {"top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
                           "chat_template_kwargs": {"enable_thinking": True}},
        })

    def test_every_sampling_file_on_main_is_accepted(self):
        for path in sorted((REPO / "configs" / "sampling").glob("*/*.yaml")):
            with self.subTest(path=path.name):
                self.assertIn("max_tokens", agent(sampling_file=path)._request_kwargs)

    def test_missing_or_unusable_sampling_is_refused(self):
        with self.assertRaisesRegex(ValueError, "needs --ak sampling_file"):
            agent(sampling_file=None)
        with self.assertRaisesRegex(ValueError, "does not send"):
            agent(sampling_file=sampling("model:\n  model_kwargs:\n    max_tokens: 10\n    seed: 1\n"))
        with self.assertRaisesRegex(ValueError, "max_tokens is required"):
            agent(sampling_file=sampling("model:\n  model_kwargs:\n    temperature: 0.6\n"))


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


class ModelNameTests(unittest.TestCase):
    def test_provider_prefixes_are_dropped_but_the_repo_is_kept(self):
        for name in ("openai/Qwen/Qwen3.5-9B", "litellm_proxy/Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B"):
            with self.subTest(name=name):
                self.assertEqual(bare_model_name(name), "Qwen/Qwen3.5-9B")


class ExtractCodeTests(unittest.TestCase):
    """extract_code: the task language's block wins, then any block; None without a block."""

    def test_block_in_the_task_language_wins_over_a_later_example(self):
        answer = "```rust\nfn f() -> i32 { 1 }\n```\nUsage:\n```bash\ncargo run\n```"
        self.assertEqual(extract_code(answer, "rust"), ("fn f() -> i32 { 1 }", "language_block"))

    def test_last_block_of_the_language_is_taken(self):
        answer = "```python\ndef f(): return 0\n```\nBetter:\n```python\ndef f(): return 1\n```"
        self.assertEqual(extract_code(answer, "python"), ("def f(): return 1", "language_block"))

    def test_common_tag_spellings_count_as_the_language(self):
        for language, tag in [("cpp", "c++"), ("cpp", "C++"), ("javascript", "js"), ("typescript", "ts"),
                              ("go", "golang"), ("python", "py"), ("ruby", "rb"), ("rust", "rs")]:
            with self.subTest(tag=tag):
                self.assertEqual(extract_code(f"```{tag}\ncode\n```", language), ("code", "language_block"))

    def test_untagged_or_other_block_is_used_when_no_language_block(self):
        self.assertEqual(extract_code("```\nint f();\n```", "cpp"), ("int f();", "other_block"))
        self.assertEqual(extract_code("```c\nint f();\n```", "cpp"), ("int f();", "other_block"))

    def test_empty_blocks_and_open_blocks_do_not_count(self):
        self.assertIsNone(extract_code("```python\n\n```", "python"))
        self.assertIsNone(extract_code("```python\ndef f():", "python"))
        self.assertIsNone(extract_code("def f(): return 1", "python"))


THINKING_OFF = "model:\n  model_kwargs:\n    max_tokens: 100\n    extra_body:\n      chat_template_kwargs:\n        enable_thinking: false\n"


class ExtractReplyTests(unittest.TestCase):
    """_extract: only the final answer counts; a cut-off reply needs a closed block."""

    def test_thinking_is_dropped(self):
        reply = "```python\ndraft\n```</think>\n```python\ndef f(): return 1\n```"
        self.assertEqual(agent()._extract(reply, "python", cut_off=False), ("def f(): return 1", "language_block"))

    def test_reply_without_a_block_is_taken_whole(self):
        reply = "plan</think>\n\nfunc f() int { return 1 }\n"
        self.assertEqual(agent()._extract(reply, "go", cut_off=False), ("func f() int { return 1 }", "whole_text"))

    def test_cut_off_while_thinking_is_not_graded(self):
        with self.assertRaisesRegex(ResponseTruncatedError, "still thinking"):
            agent()._extract("plan... ```python\ndef f(): pass\n``` more planning", "python", cut_off=True)

    def test_closed_block_after_thinking_is_graded(self):
        reply = "plan</think>\n```python\ndef f():\n    return 1\n```\nThis runs in O(1) because"
        self.assertEqual(agent()._extract(reply, "python", cut_off=True), ("def f():\n    return 1", "language_block"))

    def test_cut_off_inside_an_open_block_is_not_graded(self):
        reply = "```cpp\nint f() {\n    // Actually, let's use a different approach:\n    // Actually"
        with self.assertRaisesRegex(ResponseTruncatedError, "no complete code block"):
            agent(sampling_file=sampling(THINKING_OFF))._extract(reply, "cpp", cut_off=True)

    def test_closed_block_without_thinking_is_graded(self):
        reply = "```go\nfunc f() int { return 1 }\n```\nExplanation: the loop"
        self.assertEqual(agent(sampling_file=sampling(THINKING_OFF))._extract(reply, "go", cut_off=True),
                         ("func f() int { return 1 }", "language_block"))


class FakeClient:
    """Stands in for AsyncOpenAI: records its settings and returns one canned reply."""
    reply, finish_reason, settings, request, reasoning = "", "stop", None, None, None

    def __init__(self, **settings):
        FakeClient.settings = settings
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **request):
        FakeClient.request = request
        message = types.SimpleNamespace(content=FakeClient.reply, reasoning_content=FakeClient.reasoning)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message, finish_reason=FakeClient.finish_reason)],
            usage=types.SimpleNamespace(prompt_tokens=7, completion_tokens=100))

    async def close(self):
        pass


class FakeEnvironment:
    commands = []

    async def exec(self, command):
        self.commands.append(command)
        return types.SimpleNamespace(return_code=0, stderr="")


class RunTests(unittest.TestCase):
    def run_agent(self, a, reply, finish_reason="stop", reasoning=None):
        """Run agent `a` on a Rust task (with the locale's LANGUAGE set) against a canned reply;
        `reasoning` is what a server with a reasoning parser returns in its own field."""
        FakeClient.reply, FakeClient.finish_reason, FakeClient.reasoning = reply, finish_reason, reasoning
        env = FakeEnvironment()
        with mock.patch.object(simple_code_agent, "AsyncOpenAI", FakeClient), \
                mock.patch.dict(os.environ, {"LANGUAGE": "en_US:en"}):
            asyncio.run(a.run(instruction("rust"), env, None))
        return env

    def test_reply_is_saved_and_submitted_without_retries(self):
        a = agent()
        env = self.run_agent(a, "plan</think>\n```rust\nfn f() {}\n```")
        self.assertEqual(FakeClient.settings["max_retries"], 0)
        self.assertEqual(FakeClient.settings["timeout"], 1200.0)
        self.assertEqual(FakeClient.request["max_tokens"], 32768)
        self.assertEqual(json.loads((a.logs_dir / "solution.json").read_text()), {"language": "rust", "code": "fn f() {}"})
        self.assertEqual(json.loads((a.logs_dir / "usage.json").read_text()),
                         {"finish_reason": "stop", "prompt_tokens": 7, "completion_tokens": 100,
                          "reasoning_field": False, "code_from": "language_block"})
        self.assertIn("/workspace/solution.json", env.commands[-1])

    def test_cut_off_reply_is_saved_before_failing(self):
        a = agent()
        with self.assertRaises(ResponseTruncatedError):
            self.run_agent(a, "still planning, still planning", finish_reason="length")
        self.assertEqual((a.logs_dir / "response.txt").read_text(), "still planning, still planning")
        self.assertEqual(json.loads((a.logs_dir / "usage.json").read_text()),
                         {"finish_reason": "length", "prompt_tokens": 7, "completion_tokens": 100,
                          "reasoning_field": False})
        self.assertFalse((a.logs_dir / "solution.json").exists())

    def test_reasoning_field_is_put_back_in_front_of_the_answer(self):
        a = agent()
        self.run_agent(a, "```rust\nfn f() {}\n```", reasoning="```rust\nfn draft() {}\n```")
        self.assertEqual((a.logs_dir / "response.txt").read_text(),
                         "```rust\nfn draft() {}\n```</think>```rust\nfn f() {}\n```")
        self.assertEqual(json.loads((a.logs_dir / "solution.json").read_text())["code"], "fn f() {}")
        self.assertTrue(json.loads((a.logs_dir / "usage.json").read_text())["reasoning_field"])

    def test_cut_off_in_the_reasoning_field_is_still_thinking(self):
        a = agent(sampling_file=sampling(THINKING_OFF))  # e.g. K2-Horizon: no enable_thinking switch
        with self.assertRaisesRegex(ResponseTruncatedError, "still thinking"):
            self.run_agent(a, "", finish_reason="length", reasoning="```rust\nfn draft() {}\n``` and so")
        self.assertEqual((a.logs_dir / "response.txt").read_text(), "```rust\nfn draft() {}\n``` and so")

    def test_empty_reply_is_a_model_failure(self):
        with self.assertRaises(EmptySolutionError):
            self.run_agent(agent(), "plan</think>\n")


if __name__ == "__main__":
    unittest.main()
