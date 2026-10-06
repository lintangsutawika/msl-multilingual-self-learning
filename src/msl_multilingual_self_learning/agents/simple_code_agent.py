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


# mini-swe-agent's command for ending a task (`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`);
# models trained on its trajectories sometimes append it to a plain code answer.
SUBMIT_MARKER = re.compile(
    r"^[ \t]*echo[ \t]+['\"]?COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT['\"]?[ \t;]*$\n?",
    flags=re.MULTILINE,
)


class SimpleCodeAgent(BaseAgent):
    def __init__(self, *args, config_file: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
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
        return "0.5.0"

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
        fenced_blocks = [
            block
            for block in re.findall(
                r"```(?:[A-Za-z0-9_+#.\-]+)?\s*\n(.*?)```",
                text,
                flags=re.DOTALL,
            )
            if block.strip()
        ]

        if fenced_blocks:
            return fenced_blocks[-1].strip()

        return text

    async def run(
        self,
        instruction: str,
        environment,
        context,
    ) -> None:
        # Harbor --ae values are passed through BaseAgent.extra_env,
        # so use _get_env() rather than os.environ directly.
        language = self._get_env(
            "LANGUAGE"
        )

        match = re.match(r"Language: (python|cpp|go|java|rust|javascript|typescript|php|ruby)\n", instruction)
        if match is not None:
            task_language = match.group(1)
            if language is not None and language != task_language:
                raise ValueError("LANGUAGE does not match the task prompt")
            language = task_language
        if language not in {"python", "cpp", "go", "java", "rust", "javascript", "typescript", "php", "ruby"}:
            raise ValueError("Task must specify a supported language")

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

        # Per-request timeout in seconds (REQUEST_TIMEOUT); unset keeps the client's 600 s.
        request_timeout = self._get_env("REQUEST_TIMEOUT")
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            **({"timeout": float(request_timeout)} if request_timeout else {}),
        )

        max_tokens = int(
            self._get_env("MAX_TOKENS")
            or "8192"
        )

        # Sampling: greedy with thinking off unless set. THINKING=1 TEMPERATURE=0.6
        # TOP_P=0.95 TOP_K=20 matches the mini-swe-agent Qwen3.5 configs.
        thinking = (self._get_env("THINKING") or "0").lower() in ("1", "true", "yes")
        temperature = float(self._get_env("TEMPERATURE") or "0")
        sampling = {}
        if self._get_env("TOP_P"):
            sampling["top_p"] = float(self._get_env("TOP_P"))
        extra_body = {"chat_template_kwargs": {"enable_thinking": thinking}}
        if self._get_env("TOP_K"):
            extra_body["top_k"] = int(self._get_env("TOP_K"))

        messages = self._messages(instruction)
        # Keep the exact prompt next to the reply for review.
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "prompt.json").write_text(json.dumps(messages, ensure_ascii=False, indent=1), encoding="utf-8")

        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
            **sampling,
        )
        await client.close()
        if response.choices[0].finish_reason == "length":
            raise RuntimeError("Model response was truncated before completion")

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if content is None:
            raise RuntimeError(
                "Model returned no content"
            )

        # Keep the raw reply next to solution.json for review (what extraction removed).
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "response.txt").write_text(content, encoding="utf-8")

        code = self._strip_markdown_fences(
            content
        )

        if not code:
            raise RuntimeError(
                "Model returned empty source code"
            )

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
