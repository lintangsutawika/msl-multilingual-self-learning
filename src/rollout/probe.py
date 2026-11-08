"""The rollout job startup probe: does capture work end to end?

1. Direct SGLang call with return_prompt_token_ids / return_token_ids / logprobs:
   are prompt ids, response ids and per-token logprobs present and aligned?
2. Same kind of call through the public relay as /sessions/<trial>/v1, with a
   validate tool call + `score=` observation in the history: did the forwarder
   append a spool record (ids, logprobs, version, validate score) and strip the
   token ids from what the client received?

Prints one JSON summary to stdout; exits non-zero if any check fails.
"""

import argparse
import json
import os
import sys
import time

import httpx


def chat(base: str, key: str, model: str, messages: list, extra: dict | None = None) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": 48, "temperature": 1.0, **(extra or {})}
    r = httpx.post(f"{base}/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"{base} -> HTTP {r.status_code}: {r.text[:400]}")
    return r.json()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True)
    ap.add_argument("--api-key-file", required=True)
    ap.add_argument("--relay", required=True)
    ap.add_argument("--relay-secret", required=True)
    ap.add_argument("--spool", default="", help="spool dir; empty = capture is off, skip spool checks")
    ap.add_argument("--trial", required=True)
    a = ap.parse_args()
    key = open(a.api_key_file).read().strip()
    out: dict = {"checks": {}}
    ok = True

    models = httpx.get(f"{a.local}/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=30).json()
    model = models["data"][0]["id"]
    out["model"] = model

    # 1. Direct.
    resp = chat(a.local, key, model, [{"role": "user", "content": "Say hi in three words."}],
                {"logprobs": True, "return_prompt_token_ids": True, "return_token_ids": True})
    ch = resp["choices"][0]
    rids = ch.get("response_token_ids") or ch.get("token_ids")
    lps = (ch.get("logprobs") or {}).get("content") or []
    out["direct"] = {"choice_keys": sorted(ch.keys()), "n_prompt_ids": len(ch.get("prompt_token_ids") or []),
                     "n_response_ids": len(rids or []), "n_logprobs": len(lps)}
    out["checks"]["direct_prompt_ids"] = bool(ch.get("prompt_token_ids"))
    out["checks"]["direct_response_ids"] = bool(rids)
    out["checks"]["direct_aligned"] = bool(rids) and len(rids) == len(lps)

    # 1b. Tool calling: agents need structured tool_calls (needs --tool-call-parser).
    tools = [{"type": "function", "function": {
        "name": "terminal", "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
    tresp = chat(a.local, key, model,
                 [{"role": "user", "content": "Use the terminal tool to run `validate sub.csv`. Do not explain."}],
                 {"tools": tools, "tool_choice": "auto", "max_tokens": 2048, "temperature": 0.0})
    tmsg = tresp["choices"][0]["message"]
    tcalls = tmsg.get("tool_calls") or []
    out["tool_call"] = {"n_tool_calls": len(tcalls),
                        "first": (tcalls[0].get("function") if tcalls else None),
                        "content_head": (tmsg.get("content") or "")[:160]}
    out["checks"]["direct_tool_call_parsed"] = bool(tcalls)

    # 2. Through the relay with a session path; history contains a validate round.
    spool_file = os.path.join(a.spool or ".", f"{a.trial}.jsonl")
    before = sum(1 for _ in open(spool_file)) if os.path.exists(spool_file) else 0
    history = [
        {"role": "system", "content": "You are an ML engineer."},
        {"role": "user", "content": "Train a model and validate it."},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "terminal", "arguments": json.dumps({"command": "validate sub.csv"})}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "score=0.8731"},
        {"role": "user", "content": "Summarize the result in one sentence."},
    ]
    base = f"{a.relay.rstrip('/')}/sessions/{a.trial}"
    try:
        rresp = chat(base, a.relay_secret, model, history)
    except Exception as e:
        out["relay_tool_history_error"] = str(e)[:300]
        rresp = chat(base, a.relay_secret, model, [history[0], history[1]])
    rch = rresp["choices"][0]
    out["checks"]["relay_strips_ids"] = not any(k in rch for k in ("prompt_token_ids", "response_token_ids", "token_ids"))
    out["checks"]["relay_strips_logprobs"] = "logprobs" not in rch or rch["logprobs"] is None

    time.sleep(2)
    if not a.spool:  # serving-only endpoint (CAPTURE=0): nothing is recorded
        out["ok"] = all(out["checks"].values())
        print(json.dumps(out, indent=2))
        return 0 if out["ok"] else 1
    lines = open(spool_file).read().splitlines() if os.path.exists(spool_file) else []
    out["spool_file"] = spool_file
    out["checks"]["spool_appended"] = len(lines) > before
    if len(lines) > before:
        rec = json.loads(lines[-1])
        out["spool_record"] = {k: rec.get(k) for k in ("trial", "turn", "weight_version", "ids_logprobs_aligned",
                                                       "observed_validate_scores", "calls_validate", "finish_reason")}
        out["spool_record"]["n_prompt_ids"] = len(rec.get("prompt_token_ids") or [])
        out["spool_record"]["n_response_ids"] = len(rec.get("response_token_ids") or [])
        out["checks"]["spool_aligned"] = bool(rec.get("ids_logprobs_aligned"))
        out["checks"]["spool_version"] = rec.get("weight_version") is not None
        if "relay_tool_history_error" not in out:
            out["checks"]["spool_score_parsed"] = rec.get("observed_validate_scores") == ["0.8731"]

    ok = all(out["checks"].values())
    out["ok"] = ok
    print(json.dumps(out, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
