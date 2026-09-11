from __future__ import annotations

import base64
import json
import re

from harbor.agents.base import BaseAgent
from openai import AsyncOpenAI


class SimpleCodeAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "simple-code-agent"

    def version(self) -> str:
        return "0.4.0"

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

        match = re.fullmatch(
            r"```(?:[A-Za-z0-9_+#.\-]+)?\s*\n(.*?)```",
            text,
            flags=re.DOTALL,
        )

        if match is not None:
            return match.group(1).strip()

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

        match = re.match(r"Language: (python|cpp|go|java)\n", instruction)
        if match is not None:
            task_language = match.group(1)
            if language is not None and language != task_language:
                raise ValueError("LANGUAGE does not match the task prompt")
            language = task_language
        if language not in {"python", "cpp", "go", "java"}:
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

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a coding assistant. "
                        "Return only the complete source code "
                        "for the requested solution. "
                        "Do not include Markdown fences or explanations."
                    ),
                },
                {
                    "role": "user",
                    "content": instruction,
                },
            ],
            temperature=0,
            max_tokens=8192,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            },
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
