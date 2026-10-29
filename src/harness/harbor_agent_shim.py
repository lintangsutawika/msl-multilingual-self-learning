"""Shim over MILES's ``harbor_agent_function`` wiring OpenHands + SAIL as the rollout path.

Two jobs, both inside the MILES rollout worker (``src/`` is on its PYTHONPATH; this module
is importable as ``src.harness.harbor_agent_shim`` — no patch to miles.sif):

1. **OpenHands harness**: register an ``openhands-sdk`` HarnessBinding (LLM_* env +
   ``LITELLM_EXTRA_BODY`` logprobs/return_token_ids + ``EPISODE_CHECKER_MODULE``) and
   make ``AgentName.OPENHANDS_SDK`` resolve to our ``LiveRolloutOpenHandsSDK`` via the
   ``src.harness.backend`` factory swap. MILES picks the agent from
   ``metadata.agent_name``; set it to ``openhands-sdk`` in the seed.jsonl rows.

2. **SAIL sandbox**: MILES's ``_environment_config`` builds ``EnvironmentConfig`` with
   ``type=EnvironmentType`` (daytona/e2b/modal/singularity) — SAIL
   (``sail.harbor:SailboxEnvironment``) is a custom env only expressible via
   ``EnvironmentConfig.import_path``. We override ``_environment_config`` to emit
   ``import_path="sail.harbor:SailboxEnvironment"`` so trials run on our proven SAIL
   backend. ``sail`` must be importable on the worker PYTHONPATH (train.sh binds the MSL
   venv's site-packages) and ``SAIL_API_KEY`` set there.
"""

from __future__ import annotations

import json
import os
from typing import Any

from harbor_agent_function import HarnessBinding, HARNESS_BINDINGS, build_trial_config, run  # noqa: F401


def _openhands_env(session_url: str, api_key: str) -> dict[str, str]:
    """Env the OpenHands ``LiveRolloutOpenHandsSDK`` agent reads (OpenAI-compatible).

    ``session_url`` is MILES's node-private session-server URL (e.g.
    ``http://hnodeXXX:30021/sessions/<id>/v1``) — unreachable from a SAIL cloud
    sandbox. When ``OPENHANDS_LLM_BASE`` is set (a Modal-relay HTTPS URL that
    bridges back to the node), the agent uses THAT instead so the Sailbox can reach
    the served model. The node-side relay bridge must forward to the node's
    session server (VLLM_LOCAL_URL=http://<node>:30021).
    """
    # Route the agent's LLM call through the Modal relay so a SAIL cloud sandbox can
    # reach the node-private session server. Rewrite session_url's host to the relay
    # while PRESERVING MILES's /sessions/<id>/v1 path (per-session token accounting).
    #
    #   session_url  http://hnodeXXX:30021/sessions/<id>/v1
    #   -> LLLM_BASE https://<relay>/sessions/<id>/v1
    # The relay + node bridge forward the full /sessions/<id>/v1/* path to
    # hnodeXXX:30021. If OPENHANDS_LLM_BASE is a full URL it wins; else if it is a
    # relay origin (scheme+host), the session path is appended.
    import re as _re

    relay = os.environ.get("OPENHANDS_RELAY")
    _m = _re.search(r"/sessions/[A-Za-z0-9._-]+/v1", session_url)
    base = session_url
    if relay:
        # Preserve the /sessions/<id>/v1 suffix (or /v1) under the relay origin.
        path = _m.group(0) if _m else "/v1"
        base = relay.rstrip("/") + ("/" + path.lstrip("/") if path else "/v1")
    elif _m:
        # No relay: but still normalize only if a path exists; keep node URL as-is.
        base = session_url
    return {
        "LLM_BASE_URL": base,
        "LLM_API_KEY": api_key,
        "OPENAI_BASE_URL": base,
        "OPENAI_API_KEY": api_key,
        "LITELLM_EXTRA_BODY": '{"return_token_ids": true, "logprobs": true}',
        "EPISODE_CHECKER_MODULE": "src.episode.mlebench",
    }


OPENHANDS_BINDING = HarnessBinding(
    kwargs=lambda *a, **k: {},
    env=_openhands_env,
    model_name=lambda model: model,
)
HARNESS_BINDINGS.setdefault("openhands-sdk", OPENHANDS_BINDING)


def _sail_environment_config():
    """SAIL environment config, replacing MILES's EnvironmentType-bound builder.

    Returns an ``EnvironmentConfig`` that resolves to ``sail.harbor:SailboxEnvironment``
    via ``import_path`` (the same mechanism harbor's CLI ``--env`` uses). kwargs ride in
    ``HARBOR_ENV_KWARGS`` (JSON) as with the stock builder; the SAIL key is passed as an
    env entry so the SailboxEnvironment implementation can pick it up.
    """
    from harbor.models.trial.config import EnvironmentConfig

    kwargs = json.loads(os.getenv("HARBOR_ENV_KWARGS", "{}") or "{}")
    env: dict[str, str] = {}
    if sail_key := os.getenv("SAIL_API_KEY", ""):
        env["SAIL_API_KEY"] = sail_key
    # Raise the SAIL sandbox disk ceiling so the (large) openhands-sdk venv +
    # task archives fit; the default 20GiB overflowed -> "No space left on device"
    # -> run_agent exits 2 with no trajectory -> AgentError. The env factory
    # reads EnvironmentConfig.override_storage_mb -> task_env_config.storage_mb
    # -> SAIL disk_gib.
    return EnvironmentConfig(
        import_path="sail.harbor:SailboxEnvironment",
        env=env,
        kwargs=kwargs,
        override_storage_mb=int(os.getenv("SAIL_STORAGE_MB", "102400")),  # 100 GiB
        override_memory_mb=int(os.getenv("SAIL_MEMORY_MB", "32768")),      # 32 GiB
    )


# Replace MILES's environment builder with the SAIL one. build_trial_config calls
# ``_environment_config`` (a module-level function it imports from this module's parent
# namespace) — by shadowing it here under the same name and rebinding the module's copy,
# trials use SAIL.
_ORIG_ENV = None
try:
    import harbor_agent_function as _haf
    _ORIG_ENV = _haf._environment_config
    _haf._environment_config = _sail_environment_config
except Exception:
    pass


def _install_agent_swap() -> None:
    """Ensure ``openhands-sdk`` resolves to our LiveRolloutOpenHandsSDK (idempotent)."""
    from src.harness.backend import _install_agent_swap as _backend_swap

    _backend_swap()


_install_agent_swap()