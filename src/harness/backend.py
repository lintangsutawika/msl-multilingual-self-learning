"""Backend wiring: swap the OpenHands-SDK agent for our live-rollout subclass +
point it at MILES's served model. Import this before harbor resolves agents.

Mirrors MILES's ``MicroSweAgent`` factory-swap: we do NOT patch harbor's
site-packages; we override ``AgentFactory._AGENT_MAP[OPENHANDS_SDK]`` at runtime so
``-a openhands-sdk`` / ``AgentName.OPENHANDS_SDK`` resolves to our class.
"""

import os

from harbor.agents.factory import AgentFactory
from harbor.models.agent.name import AgentName

# The import path harbor's factory will resolve to our live-rollout agent.
_OUR_AGENT_IMPORT = "src.harness.your_openhands_agent:LiveRolloutOpenHandsSDK"


def _apply_factory_swap() -> None:
    AgentFactory._AGENT_MAP[AgentName.OPENHANDS_SDK] = _OUR_AGENT_IMPORT


_apply_factory_swap()


def mle_llm_env() -> dict[str, str]:
    """LLM env vars pointing the agent at MILES's served model (sglang endpoint).

    MILES serves the current weights via sglang (rollout GPUs) and publishes
    updated weights (theta_k -> theta_{k+1}) over time. The agent reaches it via
    the litellm-compatible endpoint; these are also settable per-run via sbatch
    ``--ae LLM_BASE_URL / LLM_MODEL / LLM_API_KEY``. This helper reads them from
    the environment so the swap is a single source of truth.
    """
    return {
        "LLM_BASE_URL": os.environ.get("LLM_BASE_URL", ""),
        "LLM_MODEL": os.environ.get("LLM_MODEL", ""),
        "LLM_API_KEY": os.environ.get("LLM_API_KEY", ""),
    }