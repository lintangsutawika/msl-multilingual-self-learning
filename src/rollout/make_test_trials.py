"""Seed the rollout job's spool with short TEST trials so the trainer can be tested without agents.

Each trial is a real multi-turn conversation with the rollout job's served model through
the public relay (/sessions/test-<task>__<i>/v1), so every spool record carries
real token ids + behavior logprobs. Between turns we append a synthetic
`validate` tool round with a made-up score, which closes an episode exactly the
way a real agent's validate would. Trials are named test-* so the production
rollout fn ignores them (the trainer must be run with SPOOL_INCLUDE=test- to use them).

    uv run --with httpx python src/rollout/make_test_trials.py --trials 6 --turns 4
"""
import argparse
import json
import os
import random
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.path.join(os.environ.get("MSL_STORE", "/groups/gai51740/aci18914wh/msl/runs"), "rollout", "endpoint.env"))
    ap.add_argument("--seed", default=str(REPO / "runs" / "seed" / "seed_list.jsonl"))
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()

    env = dict(line.split("=", 1) for line in Path(a.endpoint).read_text().split() if "=" in line)
    relay, secret = env["RELAY_URL"].rstrip("/"), env["RELAY_SECRET"]
    rows = [json.loads(l) for l in Path(a.seed).read_text().splitlines() if l.strip()]
    rng = random.Random(0)
    made = 0
    for i, row in enumerate(rows[: a.trials]):
        meta = row.get("metadata") or {}
        meta = json.loads(meta) if isinstance(meta, str) else meta
        task = (meta.get("instance_id") or meta.get("task") or f"task{i}").split("/")[-1]
        trial = f"test-{task}__{i:03d}"
        task_text = row["prompt"][0]["content"] if isinstance(row["prompt"], list) else str(row["prompt"])
        messages = [
            {"role": "system", "content": "You are an ML engineer solving a Kaggle task. Be brief."},
            {"role": "user", "content": task_text[:1500] + "\n\nDescribe your next step in two sentences."},
        ]
        score = rng.uniform(0.5, 0.7)
        for t in range(a.turns):
            r = httpx.post(f"{relay}/sessions/{trial}/v1/chat/completions",
                           headers={"Authorization": f"Bearer {secret}"},
                           json={"model": "qwen3.5-9B", "messages": messages,
                                 "max_tokens": a.max_tokens, "temperature": 1.0},
                           timeout=300)
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"].get("content") or ""
            made += 1
            score += rng.uniform(-0.05, 0.08)
            call_id = f"call_{t}"
            messages += [
                {"role": "assistant", "content": reply[:600],
                 "tool_calls": [{"id": call_id, "type": "function", "function": {
                     "name": "terminal", "arguments": json.dumps({"command": f"validate sub_{t}.csv"})}}]},
                {"role": "tool", "tool_call_id": call_id, "content": f"score={score:.4f}"},
                {"role": "user", "content": "Continue. Describe your next step in two sentences."},
            ]
        print(f"{trial}: {a.turns} calls")
    print(f"done: {made} calls -> {env.get('SPOOL_DIR')}")


if __name__ == "__main__":
    main()
