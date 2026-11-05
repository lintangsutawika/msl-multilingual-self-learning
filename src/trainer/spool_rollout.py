"""
spool_rollout.py — the trainer's MILES --rollout-function-path for the lifelong loop.

The trainer never generates. The rollout job's forwarder appends one record per model call to
the shared spool (runs/<rollout run>/spool/<trial>.jsonl: exact prompt/response
token ids, behavior logprobs, served weight version, validate scores seen in the
agent's new observations). This function turns *closed* episodes from that spool
into MILES Samples and blocks until a full batch (rollout_batch_size episodes) is available.

Episodes (MLE-Bench): within a trial, an episode is the run of calls between two
validate results. The call whose request first shows a ``score=`` observation
closes the episode made of the calls before it; its reward is the binary
improvement of that score over the trial's best so far (src.episode.mlebench).
Calls of a still-open episode are not eligible yet.

Each episode becomes ONE training sequence (the calls' token streams extend each
other: the next prompt is the previous prompt + response + tool output), with the
loss on the model's own tokens only. Per-call samples would re-process the whole
growing prompt for every call (~100x more tokens for the same response tokens).

Contract (LegacyRolloutFnAdapter): plain fn returning list[list[Sample]].

Env:
    SPOOL_DIR            the rollout job's spool dir (default runs/rollout/spool)
    SPOOL_STATE          the trainer's consumed-record ids (default <spool>/../trainer_consumed.json)
    SPOOL_INCLUDE        only trials whose id starts with one of these comma-separated
                         prefixes (default: all trials except probe-/test-)
    SPOOL_MAX_TOKENS     skip sequences longer than this (default 262144, the model's context)
    SPOOL_MAX_STALENESS  skip episodes with a call generated more than this many weight
                         versions behind the rollout job's served one (trainer.sh: 10;
                         versions = trainer step + 1, so this is roughly steps)
    SPOOL_POLL_SECS      poll interval while waiting for a batch (default 30)
    SPOOL_SORT           "longest" = longest sequences first (memory tests); default oldest first
    METRIC_DIRECTIONS    JSON file {competition: "lower"|"higher"}; trials of competitions
                         not listed are skipped
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


_REPO = Path(__file__).resolve().parents[2]
SPOOL_DIR = Path(os.environ.get("SPOOL_DIR", str(_REPO / "runs" / "rollout" / "spool")))
SPOOL_STATE = Path(os.environ.get("SPOOL_STATE", str(SPOOL_DIR.parent / "trainer_consumed.json")))
_INCLUDE = [p for p in os.environ.get("SPOOL_INCLUDE", "").split(",") if p]
_EXCLUDE_DEFAULT = ("probe-", "test-")
MAX_TOKENS = int(os.environ.get("SPOOL_MAX_TOKENS", "262144"))
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


def _closed_episodes(records: list[dict], lower_is_better: bool) -> list[tuple[list[dict], float]]:
    """(calls, reward) for every closed episode of a trial, in trial order.

    A call whose new tool output carries a signed score closes the episode made of the calls
    before it. A signed `score=` line can be replayed (cat of a log, a script printing an old
    result), and a replay is not a new evaluation: a score already seen in this trial is
    ignored unless the call it answers (the one whose prompt+response it extends) visibly ran
    `validate`. (validate run from inside a script is not visible, so new scores are always
    kept.)

    If one command validated several files (a batch of experiments), the episode is rewarded
    on the batch AVERAGE (1 if it beats the trial's best so far): padding a batch with weak
    runs costs reward, so batches of consistently good experiments are what pays. The trial's
    best then becomes the batch's best file, which the agent really has.

    The episode that produces a trial's first usable score is skipped: with no earlier
    best to beat its reward is always 1, so it carries no learning signal. Episodes whose
    validate produced no usable score (broken submission) keep their reward of 0.

    An episode with a call that hit the generation cap (finish_reason "length", the
    rollout job's MAX_GEN_TOKENS) gets reward 0 whatever its score: a runaway response
    (e.g. a repetition loop) must not be reinforced by a lucky validate after it.
    The score still counts toward the trial's best.
    """
    out: list[tuple[list[dict], float]] = []
    episode: list[dict] = []
    scores: list[float] = []
    seen_raw: set[str] = set()
    ordered = sorted(records, key=lambda r: r.get("ts", 0))
    for i, rec in enumerate(ordered):
        raw = rec.get("observed_validate_scores") or []
        if raw:
            fresh = any(x not in seen_raw for x in raw)
            if not fresh:
                parent = next((p for p in reversed(ordered[:i]) if _extends(p, rec)), None)
                fresh = bool(parent and parent.get("calls_validate"))
            seen_raw.update(raw)
            if fresh:
                seen = [s for s in (_to_score(x) for x in raw) if s is not None]
                reward = 0  # validate ran but produced no usable score
                if seen:
                    mean = sum(seen) / len(seen)
                    if scores:
                        best = min(scores) if lower_is_better else max(scores)
                        reward = int(mean < best if lower_is_better else mean > best)
                    scores.append(min(seen) if lower_is_better else max(seen))
                if any(c.get("finish_reason") == "length" for c in episode):
                    reward = 0
                if episode and (len(scores) != 1 or not seen):  # skip the first-score episode
                    out.append((episode, float(reward)))
                episode = []
        episode.append(rec)
    return out


def _extends(prev: dict, nxt: dict) -> bool:
    """True if nxt's prompt is prev's prompt + response + (tool output ...)."""
    head = list(prev["prompt_token_ids"]) + list(prev["response_token_ids"])
    return nxt["prompt_token_ids"][: len(head)] == head


def _segments(calls: list[dict]) -> list[list[dict]]:
    """Split an episode into runs of calls whose token streams extend each other. Each run
    becomes one training sequence; a break (e.g. the agent condensed its history) starts
    a new one."""
    segs: list[list[dict]] = []
    for c in calls:
        if segs and _extends(segs[-1][-1], c):
            segs[-1].append(c)
        else:
            segs.append([c])
    return segs


def _seq_len(seg: list[dict]) -> int:
    return len(seg[-1]["prompt_token_ids"]) + len(seg[-1]["response_token_ids"])


def _eligible(consumed: set[str]) -> list[tuple[list[dict], float]]:
    """Unconsumed (segment, reward) pairs from closed episodes, oldest first."""
    directions = _directions()
    served = _served_version()
    out: list[tuple[list[dict], float]] = []
    for f in sorted(SPOOL_DIR.glob("*.jsonl")):
        trial = f.stem
        if not _wanted(trial):
            continue
        direction = directions.get(_competition(trial))
        if direction is None:
            continue  # unknown metric direction: the improvement reward would be a guess
        records = []
        for line in f.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except Exception:
                continue  # a partially written last line
        for calls, reward in _closed_episodes(records, direction == "lower"):
            for seg in _segments(calls):
                if seg[0]["id"] in consumed:
                    continue
                if not all(c.get("ids_logprobs_aligned") and c.get("response_token_ids") for c in seg):
                    continue
                if _seq_len(seg) > MAX_TOKENS:
                    continue
                if MAX_STALENESS and served is not None:
                    versions = [c["weight_version"] for c in seg if c.get("weight_version") is not None]
                    if versions and served - min(versions) > int(MAX_STALENESS):
                        continue
                out.append((seg, reward))
    if os.environ.get("SPOOL_SORT") == "longest":  # memory tests: worst case first
        out.sort(key=lambda x: -_seq_len(x[0]))
    else:
        out.sort(key=lambda x: x[0][0].get("ts", 0))
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


def _to_sample(seg: list[dict], reward: float, rollout_id: int, index: int):
    """One sequence for a run of calls: the first call's prompt is the prompt; everything
    after it is the response, with the loss (and the recorded behavior logprobs) on the
    model's own tokens and the tool outputs between them masked out."""
    from miles.utils.types import Sample

    first, last = seg[0], seg[-1]
    p0 = len(first["prompt_token_ids"])
    tokens = list(last["prompt_token_ids"]) + list(last["response_token_ids"])
    n = len(tokens) - p0
    loss_mask = [0] * n
    logprobs = [0.0] * n
    for c in seg:
        start = len(c["prompt_token_ids"]) - p0
        for j, lp in enumerate(c["response_logprobs"]):
            loss_mask[start + j] = 1
            logprobs[start + j] = lp
    return Sample(
        group_index=index,
        index=index,
        rollout_id=rollout_id,
        prompt="",
        tokens=tokens,
        response=last.get("assistant_content") or "",
        response_length=n,
        reward=reward,
        loss_mask=loss_mask,
        rollout_log_probs=logprobs,
        status=Sample.Status.TRUNCATED if last.get("finish_reason") == "length" else Sample.Status.COMPLETED,
        metadata={"trial": first["trial"], "spool_ids": [c["id"] for c in seg], "calls": len(seg),
                  "weight_versions": sorted({c.get("weight_version") for c in seg}, key=lambda v: (v is None, v))},
    )


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
            _log(f"rollout {rollout_id}: {len(ready)}/{need} closed-episode sequences in {SPOOL_DIR}; waiting")
        time.sleep(POLL_SECS)
        waited += POLL_SECS
    batch = ready[:need]
    groups = [[_to_sample(seg, reward, rollout_id, i)] for i, (seg, reward) in enumerate(batch)]
    consumed.update(seg[0]["id"] for seg, _ in batch)
    _save_consumed(consumed)
    rewards = [r for _, r in batch]
    versions = sorted({c.get("weight_version") for seg, _ in batch for c in seg}, key=lambda v: (v is None, v))
    lens = [_seq_len(seg) for seg, _ in batch]
    truncated = sum(c.get("finish_reason") == "length" for seg, _ in batch for c in seg)
    _log(f"rollout {rollout_id}: {len(batch)} sequences ({sum(len(seg) for seg, _ in batch)} calls, "
         f"{truncated} truncated) from "
         f"{len({seg[0]['trial'] for seg, _ in batch})} trials, mean reward {sum(rewards) / len(rewards):.3f}, "
         f"tokens max {max(lens)} mean {sum(lens) // len(lens)}, weight versions {versions}, served {_served_version()}")
    return groups
