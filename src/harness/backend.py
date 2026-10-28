"""Backend wiring: swap the OpenHands-SDK agent for our live-rollout subclass.

Mirrors MHO's ``MicroSweAgent`` factory-swap (no sitecustomize, no site-packages
patch): we repoint ``AgentFactory._AGENT_MAP[AgentName.OPENHANDS_SDK]`` at our
``LiveRolloutOpenHandsSDK`` so harbor resolves ``-a openhands-sdk`` to it.

The swap is ARMED by explicitly importing an entrypoint that calls
:func:`_install_agent_swap` (e.g. ``src.harness.your_openhands_agent`` does it at
module import). This mirrors MHO's ``micro_swe_agent.py`` pattern -- importing the
module arms the swap once, idempotently.
"""

import os

from harbor.agents.factory import AgentFactory
from harbor.models.agent.name import AgentName

# The import path harbor's factory will resolve to our live-rollout agent.
_OUR_AGENT_IMPORT = "src.harness.your_openhands_agent:LiveRolloutOpenHandsSDK"


def _install_agent_swap() -> None:
    """Repoint ``AgentName.OPENHANDS_SDK`` at our LiveRolloutOpenHandsSDK (idempotent)."""
    try:
        if AgentFactory._AGENT_MAP[AgentName.OPENHANDS_SDK].endswith("LiveRolloutOpenHandsSDK"):
            return  # already swapped
    except (KeyError, AttributeError):
        pass
    AgentFactory._AGENT_MAP[AgentName.OPENHANDS_SDK] = _OUR_AGENT_IMPORT


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