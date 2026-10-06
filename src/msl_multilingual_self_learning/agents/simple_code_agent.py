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


LANGUAGES = {"python", "cpp", "go", "java", "rust", "javascript", "typescript", "php", "ruby"}
# A closed fenced block: an opening ``` (optional language tag), the code, a closing ```.
FENCED_BLOCK = re.compile(r"```(?:[A-Za-z0-9_+#.\-]+)?\s*\n(.*?)```", flags=re.DOTALL)


# The model's own failures, under their own names so a resume filter on RuntimeError
# (crashes) does not re-sample them: Harbor matches the exception's class name exactly.
class ResponseTruncatedError(Exception):
    """The reply hit max_tokens with no complete code block in its final answer."""


class EmptySolutionError(Exception):
    """The reply held no source code."""


# mini-swe-agent's command for ending a task (`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`);
# models trained on its trajectories sometimes append it to a plain code answer.
SUBMIT_MARKER = re.compile(
    r"^[ \t]*echo[ \t]+['\"]?COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT['\"]?[ \t;]*$\n?",
    flags=re.MULTILINE,
)


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
        return "0.6.0"

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

    def _strip_markdown_fences(
        self,
        text: str,
    ) -> str:
        text = text.strip()

        # Some models may emit a reasoning section even when thinking is
        # disabled. Keep only the content after the final </think> marker.
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[-1].strip()

        # Drop mini-swe-agent's "task done" command, which the model sometimes
        # appends to its code; it is not part of the solution.
        text = SUBMIT_MARKER.sub("", text).strip()

        # Prefer the final fenced code block when one is present (skipping
        # blocks left empty, e.g. one that only held the marker).
        fenced_blocks = [block for block in FENCED_BLOCK.findall(text) if block.strip()]

        if fenced_blocks:
            return fenced_blocks[-1].strip()

        return text

    def _thinking(self) -> bool:
        kwargs = (self._request_kwargs.get("extra_body") or {}).get("chat_template_kwargs") or {}
        return bool(kwargs.get("enable_thinking"))

    def _code_from_cut_off_reply(self, content: str) -> str:
        """Code from a reply that hit max_tokens: only a complete (closed) code block in the
        final answer counts. Cut off while still thinking (no </think> yet), any code is a
        draft; cut off inside an open block, the code is unfinished."""
        if self._thinking() and "</think>" not in content:
            raise ResponseTruncatedError("Reply hit max_tokens while still thinking")
        answer = content.rsplit("</think>", 1)[-1]
        if not any(block.strip() for block in FENCED_BLOCK.findall(SUBMIT_MARKER.sub("", answer))):
            raise ResponseTruncatedError("Reply hit max_tokens with no complete code block")
        return self._strip_markdown_fences(answer)

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

        # Harbor model names may include the provider prefix:
        #
        #   openai/Qwen/Qwen3.5-9B
        #
        # vLLM expects:
        #
        #   Qwen/Qwen3.5-9B
        if model_name.startswith(
            "openai/"
        ):
            model_name = model_name[
                len("openai/"):
            ]

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
        content = choice.message.content or ""

        # Keep the raw reply (cut off or not) and its length next to solution.json for review.
        (self.logs_dir / "response.txt").write_text(content, encoding="utf-8")
        usage = response.usage
        (self.logs_dir / "usage.json").write_text(json.dumps({
            "finish_reason": choice.finish_reason,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
        }), encoding="utf-8")

        if choice.finish_reason == "length":
            code = self._code_from_cut_off_reply(content)
        else:
            code = self._strip_markdown_fences(content)
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
