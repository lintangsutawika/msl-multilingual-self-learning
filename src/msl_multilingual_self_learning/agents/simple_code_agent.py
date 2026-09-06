import base64
import re
from pathlib import Path
from typing import override

from openai import AsyncOpenAI

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class SimpleCodeAgent(BaseAgent):
    @staticmethod
    @override
    def name() -> str:
        return "simple-code-agent"

    @override
    def version(self) -> str:
        return "0.1.0"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        # Nothing needs to be installed inside the task container.
        # The LLM call happens from the Harbor controller process.
        return

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if self.model_name is None:
            raise ValueError("SimpleCodeAgent requires --model/-m.")

        base_url = self._get_env("OPENAI_BASE_URL")
        api_key = self._get_env("OPENAI_API_KEY") or "dummy"

        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        # Harbor model names may look like:
        #
        #   openai/Qwen/Qwen3-Coder-30B-A3B-Instruct
        #
        # but an OpenAI-compatible vLLM server usually expects:
        #
        #   Qwen/Qwen3-Coder-30B-A3B-Instruct
        model = self.model_name
        if model.startswith("openai/"):
            model = model[len("openai/"):]

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a coding assistant. "
                        "Solve the programming problem exactly as requested. "
                        "Return only the final Python implementation."
                    ),
                },
                {
                    "role": "user",
                    "content": instruction,
                },
            ],
            temperature=0.0,
            max_tokens=4096,
            extra_body={
               "chat_template_kwargs": {
                    "enable_thinking": False,
               }
          },
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError("Model returned an empty response.")

        code = self._extract_code(content)

        # Encode the source so arbitrary quotes/newlines in generated code
        # cannot break the shell command.
        encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")

        command = (
            "mkdir -p /workspace && "
            f"echo '{encoded}' | base64 -d > /workspace/solution.py"
        )

        result = await environment.exec(command=command)

        if result.return_code != 0:
            raise RuntimeError(
                f"Failed to write solution.py: {result.stderr}"
            )

        # Save the raw model response on the host for later analysis.
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "model_response.txt").write_text(
            content,
            encoding="utf-8",
        )

        (self.logs_dir / "solution.py").write_text(
            code,
            encoding="utf-8",
        )

    @staticmethod
    def _extract_code(response: str) -> str:
        """
        Extract Python code from a Markdown fenced response.

        If the model returns plain code without fences, use the full response.
        """
        match = re.search(
            r"```(?:python)?\s*(.*?)```",
            response,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if match:
            return match.group(1).strip() + "\n"

        return response.strip() + "\n"