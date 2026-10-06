from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import yaml
from harbor.agents.base import BaseAgent
from openai import AsyncOpenAI


# Prompts for LeetCode tasks; run.sh passes another file with --ak config_file=...
DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "task" / "leetcode-simple-code-agent.yaml"
# model.model_kwargs keys of a configs/sampling/<repo>.yaml that SimpleCodeAgent understands.
SAMPLING_KEYS = {"max_tokens", "temperature", "top_p", "extra_body", "timeout", "drop_params"}


# The 9 task languages and the fence tags models use for each (```c++, ```js, ...).
LANGUAGE_TAGS = {
    "python": {"python", "py", "python3"},
    "cpp": {"cpp", "c++", "cxx", "cc"},
    "go": {"go", "golang"},
    "java": {"java"},
    "rust": {"rust", "rs"},
    "javascript": {"javascript", "js", "node", "nodejs"},
    "typescript": {"typescript", "ts"},
    "php": {"php"},
    "ruby": {"ruby", "rb"},
}
LANGUAGES = set(LANGUAGE_TAGS)
# A closed fenced block: an opening ``` with an optional tag (rest of that line ignored),
# the code, and a closing ```.
FENCED_BLOCK = re.compile(r"```[ \t]*([A-Za-z0-9_+#.\-]*)[^\n]*\n(.*?)```", flags=re.DOTALL)


PROVIDER_PREFIXES = ("openai/", "litellm_proxy/", "hosted_vllm/")


def bare_model_name(model_name: str) -> str:
    """The served model id without a litellm provider prefix."""
    for prefix in PROVIDER_PREFIXES:
        if model_name.startswith(prefix):
            return model_name[len(prefix):]
    return model_name


def extract_code(answer: str, language: str) -> tuple[str, str] | None:
    """(code, rule) from a final answer (the text after </think>):
      language_block -- the last non-empty block tagged with the task's language,
      other_block    -- else the last non-empty block with any other tag or none.
    None when the answer has no non-empty closed block."""
    blocks = [(tag.lower(), code.strip()) for tag, code in FENCED_BLOCK.findall(answer) if code.strip()]
    tagged = [code for tag, code in blocks if tag in LANGUAGE_TAGS[language]]
    if tagged:
        return tagged[-1], "language_block"
    if blocks:
        return blocks[-1][1], "other_block"
    return None


# The model's own failures, under their own names so a resume filter on RuntimeError
# (crashes) does not re-sample them: Harbor matches the exception's class name exactly.
class ResponseTruncatedError(Exception):
    """The reply hit max_tokens with no complete code block in its final answer."""


class EmptySolutionError(Exception):
    """The reply held no source code."""




class SimpleCodeAgent(BaseAgent):
    def __init__(self, *args, config_file: str | None = None, sampling_file: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_config(config_file)
        self._load_sampling(sampling_file)

    def _load_sampling(self, sampling_file: str | None) -> None:
        """Request settings from the model's sampling yaml (configs/sampling/<repo>.yaml),
        the file mini-swe-agent gets: model.model_kwargs, sent the way its client sends them."""
        if not sampling_file:
            raise ValueError("SimpleCodeAgent needs --ak sampling_file=configs/sampling/<repo>.yaml")
        path = Path(sampling_file)
        kwargs = dict(((yaml.safe_load(path.read_text()) or {}).get("model") or {}).get("model_kwargs") or {})
        unknown = set(kwargs) - SAMPLING_KEYS
        if unknown:
            raise ValueError(f"{path}: model.model_kwargs has settings SimpleCodeAgent does not send: {sorted(unknown)}")
        if "max_tokens" not in kwargs:
            raise ValueError(f"{path}: model.model_kwargs.max_tokens is required (vLLM would allow the whole context)")
        kwargs.pop("drop_params", None)  # a litellm option (mini-swe-agent's client); nothing to send
        timeout = kwargs.pop("timeout", None)
        self._client_kwargs = {"timeout": float(timeout)} if timeout is not None else {}
        self._request_kwargs = kwargs

    def _load_config(self, config_file: str | None) -> None:
        path = Path(config_file) if config_file else DEFAULT_CONFIG
        agent = (yaml.safe_load(path.read_text()) or {}).get("agent") or {}
        missing = [k for k in ("system_template", "instance_template", "drop_task_from") if not agent.get(k)]
        if missing:
            raise ValueError(f"{path} is not a SimpleCodeAgent config (missing agent.{', agent.'.join(missing)})")
        if "{{task}}" not in agent["instance_template"]:
            raise ValueError(f"{path}: agent.instance_template has no {{{{task}}}} placeholder")
        self._config = agent

    @staticmethod
    def name() -> str:
        return "simple-code-agent"

    def version(self) -> str:
        return "0.8.0"

    def _messages(self, instruction: str) -> list[dict[str, str]]:
        """System and user messages: the task text up to agent.drop_task_from (the
        submission steps written for mini-swe-agent), inside agent.instance_template."""
        marker = self._config["drop_task_from"]
        if marker not in instruction:
            raise ValueError(f"task instruction has no {marker!r}; update agent.drop_task_from in the config")
        task = instruction[: instruction.index(marker)].rstrip()
        return [
            {"role": "system", "content": self._config["system_template"].strip()},
            {"role": "user", "content": self._config["instance_template"].replace("{{task}}", task).strip()},
        ]

    async def setup(
        self,
        environment,
    ) -> None:
        """
        No additional setup is required.

        The model server runs outside the Harbor sandbox and is accessed
        through its OpenAI-compatible HTTP endpoint.
        """
        return None

    def _thinking(self) -> bool:
        kwargs = (self._request_kwargs.get("extra_body") or {}).get("chat_template_kwargs") or {}
        return bool(kwargs.get("enable_thinking"))

    @staticmethod
    def _reply_text(message) -> tuple[str, bool]:
        """(text, reasoned): the reply in one format, reasoning</think>answer. A server with a
        reasoning parser returns the reasoning in its own field (reasoning_content or
        reasoning) and only the answer as content; it is put back in front of the answer,
        with </think> only once an answer has started."""
        content = message.content or ""
        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or ""
        if not reasoning:
            return content, False
        return reasoning + ("</think>" + content if content else ""), True

    def _extract(self, content: str, language: str, cut_off: bool, thinking: bool | None = None) -> tuple[str, str]:
        """(code, rule) from the reply. Only the final answer counts (the text after the
        last </think>). A block tagged with the task's language wins over other blocks; a
        reply with no block is taken whole (rule whole_text). A reply that hit max_tokens
        needs a closed block: cut off while still thinking (no </think> yet) any code is a
        draft, and cut off inside an open block the code is unfinished."""
        thinking = self._thinking() if thinking is None else thinking
        if cut_off and thinking and "</think>" not in content:
            raise ResponseTruncatedError("Reply hit max_tokens while still thinking")
        answer = content.rsplit("</think>", 1)[-1].strip()
        found = extract_code(answer, language)
        if found:
            return found
        if cut_off:
            raise ResponseTruncatedError("Reply hit max_tokens with no complete code block")
        return answer, "whole_text"

    async def run(
        self,
        instruction: str,
        environment,
        context,
    ) -> None:
        # The task's language comes from its "Language:" line only (not the LANGUAGE
        # environment variable, which is also the locale setting, e.g. en_US:en).
        match = re.match(r"Language: (\w+)\n", instruction)
        if match is None or match.group(1) not in LANGUAGES:
            raise ValueError("Task instruction must start with 'Language: <one of the 9 languages>'")
        language = match.group(1)

        base_url = (
            self._get_env(
                "OPENAI_BASE_URL"
            )
            or "http://127.0.0.1:8000/v1"
        )

        api_key = (
            self._get_env(
                "OPENAI_API_KEY"
            )
            or "EMPTY"
        )

        model_name = (
            self._get_env(
                "MODEL_NAME"
            )
            or self.model_name
            or "Qwen/Qwen3.5-9B"
        )

        # Harbor model names may carry a litellm provider prefix (run.sh: openai/...,
        # run.sbatch: litellm_proxy/...); vLLM serves the bare repo, Qwen/Qwen3.5-9B.
        model_name = bare_model_name(model_name)

        # No client-side retries: a request past the sampling file's timeout ends as
        # APITimeoutError (re-run on resume) instead of being re-sent until Harbor's
        # agent timeout kills the trial.
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0, **self._client_kwargs)

        messages = self._messages(instruction)
        # Keep the exact prompt next to the reply for review.
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "prompt.json").write_text(json.dumps(messages, ensure_ascii=False, indent=1), encoding="utf-8")

        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            **self._request_kwargs,
        )
        await client.close()
        choice = response.choices[0]
        content, reasoned = self._reply_text(choice.message)

        # Keep the reply (cut off or not) and its length next to solution.json for review.
        (self.logs_dir / "response.txt").write_text(content, encoding="utf-8")
        usage = response.usage
        record = {
            "finish_reason": choice.finish_reason,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "reasoning_field": reasoned,
        }
        (self.logs_dir / "usage.json").write_text(json.dumps(record), encoding="utf-8")

        code, record["code_from"] = self._extract(content, language, cut_off=choice.finish_reason == "length",
                                                   thinking=self._thinking() or reasoned)
        (self.logs_dir / "usage.json").write_text(json.dumps(record), encoding="utf-8")
        if not code:
            raise EmptySolutionError("Model returned no source code")

        solution = {
            "language": language,
            "code": code,
        }

        solution_json = json.dumps(
            solution,
            ensure_ascii=False,
        )
        # Preserve the exact submission for review after Harbor tears down the
        # container. The agent's only workspace output is solution.json.
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "solution.json").write_text(solution_json, encoding="utf-8")

        encoded = base64.b64encode(
            solution_json.encode(
                "utf-8"
            )
        ).decode(
            "ascii"
        )

        command = (
            "mkdir -p /workspace && "
            f"printf '%s' '{encoded}' "
            "| base64 -d "
            "> /workspace/solution.json"
        )

        result = await environment.exec(
            command
        )

        if result.return_code != 0:
            raise RuntimeError(
                "Failed to write "
                "/workspace/solution.json: "
                f"{result.stderr}"
            )
