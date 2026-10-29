"""Custom OpenHands-SDK agent for live, MILES-trainable rollouts.

Subclasses harbor's ``OpenHandsSDK`` adapter (host-side) WITHOUT patching harbor's
site-packages. Differences from the stock agent:

- Ships OUR ``openhands_sdk_runner.py`` (this package) into the sandbox instead of
  harbor's stock runner, adding:
  * live ``/logs/agent/steps.jsonl`` (agent console stream, ongoing),
  * a poller-thread ``/logs/agent/rollout.jsonl`` (structured per-event live rollouts).
- Requests per-token **logprobs** + **token_ids** from the LLM backend
  (``LITELLM_EXTRA_BODY: {"return_token_ids": true, "logprobs": true}``) so the
  rollouts carry the sampling distribution needed for PPO importance correction
  (matching micro-swe-agent / MILES's ``_query``).

Register via the agent-factory swap (AgentName.OPENHANDS_SDK -> this class), the
same pattern MILES uses for ``MicroSweAgent``.
"""

import json
from pathlib import Path
from typing import Any, override

from harbor.agents.installed.openhands_sdk import OpenHandsSDK
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

# Our patched runner lives next to this file.
_RUNNER = Path(__file__).parent / "openhands_sdk_runner.py"


class LiveRolloutOpenHandsSDK(OpenHandsSDK):
    """OpenHands SDK agent that captures ongoing rollouts + per-token logprobs."""

    # NOTE: no `name = "..."` class attr here — the base `BaseInstalledAgent` defines
    # `name()` as a callable (used by `to_agent_info`), and a string attr would shadow
    # it -> `TypeError: 'str' object is not callable`. The agent registers under its
    # AgentName via the factory swap (openhands-sdk -> this class).

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        """Install the OpenHands SDK venv WITH mcp + litellm.

        The stock OpenHandsSDK install omits ``mcp``, and litellm (used by the SDK's
        LLM layer) imports ``mcp.types`` -> every trial dies ModuleNotFoundError: mcp.
        Mirror MHO's MicroSweAgent fix: add mcp (+ litellm) to the sandbox pip install.
        """
        # Check if already installed (mirror base OpenHandsSDK.install -- no is_installed
        # method; it probes the sandbox python directly).
        check_result = await environment.exec(
            command='[ -f /opt/openhands-sdk-venv/bin/python ] && /opt/openhands-sdk-venv/bin/python -c "import openhands.sdk" 2>/dev/null',
        )
        if check_result.return_code == 0:
            return
        agent_user = environment.default_user or "root"
        await self.exec_as_root(
            environment,
            command=f"mkdir -p /opt/openhands-sdk-venv && chown {agent_user}:{agent_user} /opt/openhands-sdk-venv",
        )
        version_spec = f"=={self._version}" if self._version else ""
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "curl -LsSf https://astral.sh/uv/install.sh | sh && "
                'if [ -f "$HOME/.local/bin/env" ]; then source "$HOME/.local/bin/env"; fi && '
                f"uv python install {self._python_version} && "
                f"uv venv /opt/openhands-sdk-venv --python {self._python_version} --clear && "
                "source /opt/openhands-sdk-venv/bin/activate && "
                f"uv pip install openhands-sdk{version_spec} openhands-tools{version_spec} fastapi mcp litellm"
            ),
        )

    @override
    async def configure(self, environment: BaseEnvironment) -> None:
        """Upload OUR runner (with live rollout + logprob capture), not stock."""
        if not _RUNNER.exists():
            raise FileNotFoundError(f"Our runner missing: {_RUNNER}")
        local_copy = self.logs_dir / "run_agent.py"
        local_copy.write_text(_RUNNER.read_text())
        await environment.upload_file(
            source_path=local_copy,
            target_path="/installed-agent/run_agent.py",
        )
        await environment.exec(
            command="chmod +x /installed-agent/run_agent.py",
            user="root",
        )

    @override
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        """Run with per-token logprobs + token_ids requested from the LLM backend."""
        # Delegate to the base run() but inject the RL rollout params via env
        # that base sets. We can't easily reach inside base.run(), so build the
        # env we need by calling base.setup + the stock run with an extra flag.
        # Simplest: replicate the base.run() env assembly, adding logprobs.
        import shlex

        escaped_instruction = shlex.quote(instruction)
        env: dict[str, str] = {}

        llm_api_key = self._get_env("LLM_API_KEY")
        if llm_api_key is None:
            raise ValueError("LLM_API_KEY environment variable must be set")
        env["LLM_API_KEY"] = llm_api_key

        llm_base_url = self._get_env("LLM_BASE_URL")
        if llm_base_url is not None:
            env["LLM_BASE_URL"] = llm_base_url

        if self.model_name:
            env["LLM_MODEL"] = self.model_name
        elif (llm_model := self._get_env("LLM_MODEL")) is not None:
            env["LLM_MODEL"] = llm_model
        else:
            raise ValueError("No LLM model specified")

        env["AGENT_LOGS_DIR"] = "/logs/agent"
        env["TRAJECTORY_PATH"] = f"/logs/agent/{self._TRAJECTORY_FILENAME}"
        env["LOAD_SKILLS"] = "1" if self._load_skills else "0"
        env["SKILL_PATHS"] = ":".join(self._skill_paths)
        # SDK-native persistence/resume: a stable persistence dir + conversation_id lets
        # the SDK save/restore the full conversation state so a restarted trial resumes
        # from its prior trajectory. Per-trial dir on shared FS; stable id per trial.
        _trials_dir = self._get_env("HARBOR_TRIALS_DIR")
        _persist = self._get_env("PERSISTENCE_DIR")
        if _persist:
            env["PERSISTENCE_DIR"] = _persist
        elif _trials_dir:
            env["PERSISTENCE_DIR"] = f"{_trials_dir}/conversations"
        _conv_id = self._get_env("CONVERSATION_ID")
        if _conv_id:
            env["CONVERSATION_ID"] = _conv_id

        if self.mcp_servers:
            mcp_list: list[dict[str, str | list[str]]] = []
            for server in self.mcp_servers:
                entry: dict[str, str | list[str]] = {
                    "name": server.name,
                    "transport": server.transport,
                }
                if server.transport == "stdio":
                    if server.command:
                        entry["command"] = server.command
                    if server.args:
                        entry["args"] = server.args
                else:
                    if server.url:
                        entry["url"] = server.url
                mcp_list.append(entry)
            env["MCP_SERVERS_JSON"] = json.dumps(mcp_list)

        # RL rollout data: return_token_ids + logprobs (per-token sampling probs).
        env["LITELLM_EXTRA_BODY"] = json.dumps(
            {"return_token_ids": True, "logprobs": True}
        )
        # Pass a native SDK LLM config asking litellm for logprobs as well.
        env["LLM_CONFIG"] = json.dumps({"litellm_extra_body": {"logprobs": True}})

        if self._max_iterations is not None:
            env["MAX_ITERATIONS"] = str(self._max_iterations)
        # Per-sample episode control: cap each run to ~one validate-round
        # (MAX_VALIDATE_ROUND → the runner stops after ~one validate), so a MILES
        # sample = one episode. 0/unset = full solve per sample.
        if (vri := self._get_env("VALIDATE_ROUND_ITERS")):
            env["VALIDATE_ROUND_ITERS"] = vri
        if self._temperature is not None:
            env["LLM_TEMPERATURE"] = str(self._temperature)

        # Run OUR runner (live steps + poller rollout.jsonl) with stdout captured.
        command = f"""
/opt/openhands-sdk-venv/bin/python /installed-agent/run_agent.py \
    --instruction={escaped_instruction} \
    --logs-dir="$AGENT_LOGS_DIR" \
    --trajectory-path="$TRAJECTORY_PATH" \
    > /logs/agent/steps.jsonl 2>&1 </dev/null
"""
        await self.exec_as_agent(environment, command=command.strip(), env=env)


# Alias for the factory swap.
OpenHandsSDKLive = LiveRolloutOpenHandsSDK


# Arm the AgentFactory swap on import (mirrors MHO's micro_swe_agent.py): make
# AgentName.OPENHANDS_SDK resolve to this class so harbor "-a openhands-sdk" (eval)
# and MILES's in-process agent-name lookup use LiveRolloutOpenHandsSDK. Importing
# this module arms it; idempotent.
from .backend import _install_agent_swap  # noqa: E402

_install_agent_swap()

