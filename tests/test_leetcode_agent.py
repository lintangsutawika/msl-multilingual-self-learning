import base64
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from msl_multilingual_self_learning.agents.simple_code_agent import SimpleCodeAgent
from src.benchmarks.leetcode.adapter import LANGUAGES, generate
from src.benchmarks.leetcode.dataset import DEFAULT_DATASET, load_problem
from src.benchmarks.leetcode.prompt import build_prompt


class PromptTests(unittest.TestCase):
    def test_all_languages_share_problem_and_instructions(self):
        record = load_problem(DEFAULT_DATASET, 3243)
        prompts = [build_prompt(record, language) for language in LANGUAGES]
        self.assertEqual(len({prompt.split('Problem:\n', 1)[1] for prompt in prompts}), 1)
        for language, prompt in zip(LANGUAGES, prompts):
            self.assertTrue(prompt.startswith(f'Language: {language}\n'))
            self.assertIn(record['interfaces'][language]['raw_signature'], prompt)
            self.assertNotIn('starter', prompt.lower())
            self.assertNotIn('check(candidate)', prompt)

    def test_only_python_judging_and_separate_native_adapters(self):
        record = load_problem(DEFAULT_DATASET, 3243)
        with tempfile.TemporaryDirectory() as directory:
            tasks = [generate(record, language, Path(directory)) for language in LANGUAGES]
            self.assertEqual(len({(task / 'tests/test.py').read_bytes() for task in tasks}), 1)
            self.assertEqual(len({(task / 'tests/canonical_test.py').read_bytes() for task in tasks}), 1)
            for task in tasks:
                self.assertEqual({p.name for p in (task / 'tests').iterdir()},
                                 {'test.py', 'canonical_test.py', 'config.json', 'test.sh', '_package_submission.py'})
                self.assertTrue(any((task / 'environment/files/adapters').iterdir()))
                self.assertFalse((task / 'solution/solution.json').exists())


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_mixed_language_tasks_write_only_solution_json(self):
        for language in LANGUAGES:
            with self.subTest(language=language), tempfile.TemporaryDirectory() as directory:
                code = 'complete source code'
                response = SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=code))])
                client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response))), close=AsyncMock())
                environment = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stderr='')))
                agent = SimpleCodeAgent(logs_dir=Path(directory), model_name='openai/Qwen/Qwen3.5-9B')
                with patch('msl_multilingual_self_learning.agents.simple_code_agent.AsyncOpenAI', return_value=client):
                    await agent.run(f'Language: {language}\nEntrypoint: f\n', environment, None)
                command = environment.exec.call_args.args[0]
                encoded = command.split("printf '%s' '")[1].split("'")[0]
                submission = json.loads(base64.b64decode(encoded))
                self.assertEqual(submission, {'language': language, 'code': code})
                self.assertTrue(command.endswith('> /workspace/solution.json'))
                self.assertEqual(json.loads((Path(directory) / 'solution.json').read_text()), submission)
                self.assertEqual(client.chat.completions.create.call_args.kwargs['model'], 'Qwen/Qwen3.5-9B')

    async def test_mismatched_language_override_fails_before_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = SimpleCodeAgent(logs_dir=Path(directory), extra_env={'LANGUAGE': 'cpp'})
            with self.assertRaisesRegex(ValueError, 'does not match'):
                await agent.run('Language: go\n', None, None)


if __name__ == '__main__':
    unittest.main()
