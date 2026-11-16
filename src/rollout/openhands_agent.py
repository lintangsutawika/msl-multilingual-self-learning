"""Harbor's stock openhands-sdk agent plus an LLM-summarizing context condenser.

Harbor's runner builds ``Agent(llm, tools, agent_context)`` without a condenser, so a
long trial's prompt grows until it hits the model's context (MLE-Bench trials reach
150k+ tokens). This agent uploads the same runner with one addition: when
CONDENSER_MAX_EVENTS is set (an agent env, ``--ae``), the agent gets OpenHands'
``LLMSummarizingCondenser``, which replaces older history with an LLM-written summary
once the conversation exceeds that many events, keeping the first CONDENSER_KEEP_FIRST
events (system prompt + task). The summary call goes through the same relay session,
so it is recorded in the spool like any other call.

When TERMINAL_MAX_TIMEOUT is set, no terminal action may block longer than that many
seconds, whatever timeout the model asked for.

    harbor run -a src.rollout.openhands_agent:CondensingOpenHandsSDK \\
        --ae CONDENSER_MAX_EVENTS=120 ...
"""
from pathlib import Path

from harbor.agents.installed import openhands_sdk as _sdk
from harbor.agents.installed.openhands_sdk import OpenHandsSDK

_HOOK = "    agent = Agent(**agent_kwargs)\n"
_CONDENSER = """\
    # msl: summarize older history past CONDENSER_MAX_EVENTS events so long runs stay
    # within the model's (and the trainer's) context.
    if os.environ.get("CONDENSER_MAX_EVENTS"):
        from openhands.sdk.context.condenser import LLMSummarizingCondenser

        agent_kwargs["condenser"] = LLMSummarizingCondenser(
            llm=llm.model_copy(update={"usage_id": "condenser"}),
            max_size=int(os.environ["CONDENSER_MAX_EVENTS"]),
            keep_first=int(os.environ.get("CONDENSER_KEEP_FIRST", "4")),
        )
        print(f"Condenser: max {os.environ['CONDENSER_MAX_EVENTS']} events")

    # msl: cap how long one terminal action may block at TERMINAL_MAX_TIMEOUT seconds. The
    # agent learned to poke a stuck shell (C-c, empty input, `echo ''`) with 3600-10000s
    # timeouts, idling a trial for hours (2026-11-05). OpenHands' own cap
    # (OH_RUNTIME_IDLE_TIMEOUT_SECONDS) skips exactly those inputs, so clamp every action
    # here. Only execution is capped; the model's tool call stays as generated.
    if os.environ.get("TERMINAL_MAX_TIMEOUT"):
        from openhands.tools.terminal.impl import TerminalExecutor

        _cap = float(os.environ["TERMINAL_MAX_TIMEOUT"])
        _run = TerminalExecutor.__call__

        def _capped(self, action, conversation=None):
            if action.timeout is not None and action.timeout > _cap:
                action = action.model_copy(update={"timeout": _cap})
            return _run(self, action, conversation)

        TerminalExecutor.__call__ = _capped
        print(f"Terminal timeout: max {_cap:g}s")
"""


def runner_source() -> str:
    text = (Path(_sdk.__file__).parent / "openhands_sdk_runner.py").read_text()
    if _HOOK not in text:
        raise RuntimeError("harbor's openhands_sdk_runner.py changed: no `agent = Agent(**agent_kwargs)` line")
    return text.replace(_HOOK, _CONDENSER + _HOOK, 1)


class CondensingOpenHandsSDK(OpenHandsSDK):
    async def install(self, environment) -> None:
        await super().install(environment)
        local = self.logs_dir / "run_agent.py"
        local.write_text(runner_source())
        await environment.upload_file(source_path=local, target_path="/installed-agent/run_agent.py")
        await environment.exec(command="chmod +x /installed-agent/run_agent.py", user="root")
