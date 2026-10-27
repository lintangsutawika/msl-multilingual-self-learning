"""MLE-Bench episode definition: boundary + per-episode reward.

The episode rule for MLE-Bench is DOMAIN knowledge about the benchmark, not about any
harness — so it lives here, under ``src/episode/``, importable by both:

  - the OpenHands runner (via ``EPISODE_CHECKER_MODULE=src.episode.mlebench``) to decide
    when a live rollout's episode ends, and
  - the harbor task verifier (``tests/test.sh``) to emit the per-episode reward.

Episode semantics (MLE-Bench, continuous self-learning):
  An episode is the span of agent steps between two ``validate`` calls — solve-start ->
  first validate, then validate -> validate. The agent iterates (edit -> validate -> ...),
  and each validate ActionEvent closes the episode that preceded it.

  This matches notes.tex telescoping rewards: reward per episode = binary improvement in
  the validation score, so a long-horizon PPO update happens at each validate boundary
  (rather than one terminal reward for the whole run).
"""

from __future__ import annotations

from typing import Any


def is_validate_action(event: Any) -> bool:
    """True if ``event`` is an agent tool action invoking the ``validate`` tool."""
    tc = getattr(event, "tool_call", None)
    fn = getattr(tc, "function", None) if tc is not None else None
    name = getattr(fn, "name", None) if fn is not None else getattr(event, "tool_name", None)
    return name == "validate" or getattr(event, "tool_name", None) == "validate"


def check_episode_completion(event: Any, state: dict) -> bool:
    """Episode-completion hook for the live rollout runner.

    Returns True when ``event`` closes the current MLE-Bench episode (a ``validate``
    action). The runner then seals the episode (writes ``episode_end`` to rollout.jsonl)
    and bumps ``state["episode_index"]`` / ``state["start_event"]``.
    """
    # Only tool actions can be a validate boundary.
    if "Action" not in type(event).__name__ and not hasattr(event, "tool_call"):
        return False
    return is_validate_action(event)


def compute_episode_reward(
    submission_scores: list[float],
    *,
    prior_best: float | None = None,
    margin: float = 1e-12,
) -> list[int]:
    """Per-episode binary-improvement reward for a sequence of validate scores.

    Each validate call is an episode end. Reward = 1 if that submit's score beats the
    running best so far (strictly lower for MLE-Bench, where lower is better), else 0.
    This is the notes.tex telescoping reward used for long-horizon PPO.

    Args:
        submission_scores: scores in episode order (one per validate call). Lower is better.
        prior_best: score to seed the running best (e.g. an earlier run's best). If None,
            the first submission sets the baseline -> gets reward 0 (no improvement over
            an empty prior) only if prior_best is None and it's the first; otherwise the
            first improvement over prior_best gets +1.
        margin: epsilon so a tie is not an improvement.

    Returns:
        A list of 0/1 rewards, one per episode (same length as submission_scores).
    """
    rewards: list[int] = []
    best = prior_best
    for score in submission_scores:
        improved = best is None or (score + margin) < best
        rewards.append(1 if improved else 0)
        if improved:
            best = score
    return rewards