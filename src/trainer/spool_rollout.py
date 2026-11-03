"""
spool_rollout.py — the trainer's MILES --rollout-function-path for the lifelong loop.

The trainer never generates. The rollout job's forwarder appends one record per model call to
the shared spool (runs/<rollout run>/spool/<trial>.jsonl: exact prompt/response
token ids, behavior logprobs, served weight version, validate scores seen in the
agent's new observations). This function turns *closed* episodes from that spool
into MILES Samples and blocks until a full batch is available.

Episodes (MLE-Bench): within a trial, an episode is the run of calls between two
validate results. The call whose request first shows a ``score=`` observation
closes the episode made of the calls before it; its reward is the binary
improvement of that score over the trial's best so far (src.episode.mlebench).
Calls of a still-open episode are not eligible yet.

Contract (LegacyRolloutFnAdapter): plain fn returning list[list[Sample]].

Env:
    SPOOL_DIR            the rollout job's spool dir (default runs/rollout/spool)
    SPOOL_STATE          the trainer's consumed-record ids (default <spool>/../trainer_consumed.json)
    SPOOL_INCLUDE        only trials whose id starts with one of these comma-separated
                         prefixes (default: all trials except probe-/test-)
    SPOOL_MAX_TOKENS     skip calls whose prompt+response exceed this (default 16384)
    SPOOL_MAX_STALENESS  skip calls generated more than this many weight versions
                         behind the rollout job's current one (default: no limit)
    SPOOL_POLL_SECS      poll interval while waiting for a batch (default 30)
    METRIC_DIRECTIONS    JSON file {competition: "lower"|"higher"}; default "higher"
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from src.episode.mlebench import compute_episode_reward

_REPO = Path(__file__).resolve().parents[2]
SPOOL_DIR = Path(os.environ.get("SPOOL_DIR", str(_REPO / "runs" / "rollout" / "spool")))
SPOOL_STATE = Path(os.environ.get("SPOOL_STATE", str(SPOOL_DIR.parent / "trainer_consumed.json")))
_INCLUDE = [p for p in os.environ.get("SPOOL_INCLUDE", "").split(",") if p]
_EXCLUDE_DEFAULT = ("probe-", "test-")
MAX_TOKENS = int(os.environ.get("SPOOL_MAX_TOKENS", "16384"))
MAX_STALENESS = os.environ.get("SPOOL_MAX_STALENESS")
POLL_SECS = float(os.environ.get("SPOOL_POLL_SECS", "30"))


def _log(msg: str) -> None:
    print(f"[spool_rollout] {msg}", flush=True)


def _wanted(trial: str) -> bool:
    if _INCLUDE:
        return any(trial.startswith(p) for p in _INCLUDE)
    return not trial.startswith(_EXCLUDE_DEFAULT) and trial != "_untagged"


def _competition(trial: str) -> str:
    name = trial.split("__")[0]
    for p in _INCLUDE + list(_EXCLUDE_DEFAULT):
        if name.startswith(p):
            name = name[len(p):]
    return name


def _directions() -> dict:
    path = os.environ.get("METRIC_DIRECTIONS")
    if path and os.path.exists(path):
        return json.load(open(path))
    return {}


def _served_version() -> int | None:
    try:
        return int((SPOOL_DIR.parent / "weight_version").read_text().strip())
    except Exception:
        return None


def _to_score(raw: str) -> float | None:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if v == v and v not in (float("inf"), float("-inf")) else None


def _closed_episode_calls(records: list[dict], lower_is_better: bool) -> list[tuple[dict, float]]:
    """(call, episode reward) for every call in a closed episode, in trial order."""
    out: list[tuple[dict, float]] = []
    episode: list[dict] = []
    scores: list[float] = []
    for rec in sorted(records, key=lambda r: r.get("ts", 0)):
        if rec.get("observed_validate_scores"):
            seen = [s for s in (_to_score(x) for x in rec["observed_validate_scores"]) if s is not None]
            reward = 0  # validate ran but produced no usable score
            if seen:
                scores.append(seen[-1])
                reward = compute_episode_reward(scores, lower_is_better=lower_is_better)[-1]
            out.extend((c, float(reward)) for c in episode)
            episode = []
        episode.append(rec)
    return out


def _eligible(consumed: set[str]) -> list[tuple[dict, float]]:
    directions = _directions()
    served = _served_version()
    out: list[tuple[dict, float]] = []
    for f in sorted(SPOOL_DIR.glob("*.jsonl")):
        trial = f.stem
        if not _wanted(trial):
            continue
        records = []
        for line in f.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except Exception:
                continue  # a partially written last line
        lower = directions.get(_competition(trial), "higher") == "lower"
        for rec, reward in _closed_episode_calls(records, lower):
            if rec["id"] in consumed or not rec.get("ids_logprobs_aligned"):
                continue
            n = len(rec.get("prompt_token_ids") or []) + len(rec.get("response_token_ids") or [])
            if not rec.get("response_token_ids") or n > MAX_TOKENS:
                continue
            if MAX_STALENESS and served is not None and rec.get("weight_version") is not None:
                if served - rec["weight_version"] > int(MAX_STALENESS):
                    continue
            out.append((rec, reward))
    out.sort(key=lambda x: x[0].get("ts", 0))  # oldest first
    return out


def _load_consumed() -> set[str]:
    try:
        return set(json.loads(SPOOL_STATE.read_text()))
    except Exception:
        return set()


def _save_consumed(consumed: set[str]) -> None:
    tmp = SPOOL_STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(sorted(consumed)))
    os.replace(tmp, SPOOL_STATE)


def _to_sample(rec: dict, reward: float, rollout_id: int, index: int):
    from miles.utils.types import Sample

    prompt_ids = rec["prompt_token_ids"]
    response_ids = rec["response_token_ids"]
    s = Sample(
        group_index=index,
        index=index,
        rollout_id=rollout_id,
        prompt="",
        tokens=list(prompt_ids) + list(response_ids),
        response=rec.get("assistant_content") or "",
        response_length=len(response_ids),
        reward=reward,
        loss_mask=[1] * len(response_ids),
        rollout_log_probs=list(rec["response_logprobs"]),
        status=Sample.Status.TRUNCATED if rec.get("finish_reason") == "length" else Sample.Status.COMPLETED,
        metadata={"trial": rec["trial"], "turn": rec.get("turn"), "spool_id": rec["id"],
                  "weight_version": rec.get("weight_version")},
    )
    return s


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    """Plain fn form; wrapped by LegacyRolloutFnAdapter. Returns list[list[Sample]]."""
    if evaluation:
        return []
    need = int(getattr(args, "rollout_batch_size", 8))
    consumed = _load_consumed()
    waited = 0.0
    while True:
        ready = _eligible(consumed)
        if len(ready) >= need:
            break
        if waited % 300 < POLL_SECS:
            _log(f"rollout {rollout_id}: {len(ready)}/{need} closed-episode calls in {SPOOL_DIR}; waiting")
        time.sleep(POLL_SECS)
        waited += POLL_SECS
    batch = ready[:need]
    groups = [[_to_sample(rec, reward, rollout_id, i)] for i, (rec, reward) in enumerate(batch)]
    consumed.update(rec["id"] for rec, _ in batch)
    _save_consumed(consumed)
    rewards = [r for _, r in batch]
    versions = sorted({rec.get("weight_version") for rec, _ in batch}, key=lambda v: (v is None, v))
    _log(f"rollout {rollout_id}: {len(batch)} samples from {len({rec['trial'] for rec, _ in batch})} trials, "
         f"mean reward {sum(rewards) / len(rewards):.3f}, weight versions {versions}, served {_served_version()}")
    return groups
