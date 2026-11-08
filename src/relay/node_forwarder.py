"""
Node-side bridge for the Modal reverse relay (harbor/relay/public_api.py).

Runs on the vLLM node (same Slurm job as the server). Dials OUT to the relay's
WebSocket and holds it open; for each request frame the relay forwards, it calls
the local vLLM and sends the response back. This is the leg that makes a private
node reachable from Modal without accepting any inbound connection.

Env:
    RELAY_WS_URL    wss://<workspace>--tts-vllm-relay-web.modal.run/bridge
    RELAY_SECRET    shared secret (matches the Modal 'tts-relay' secret)
    VLLM_LOCAL_URL  local vLLM base, e.g. http://localhost:8000  (no trailing /v1)
    CAPTURE_DIR     (optional) spool dir; when set, every chat completion is recorded
                    there for the trainer (see "Rollout capture" below)
    WEIGHT_VERSION_FILE  (optional) file holding the served weight version (int)
    MAX_GEN_TOKENS  (optional) cap on the tokens one chat completion may generate
                    (0/unset = no cap; agents send no limit of their own)

Run (deps resolved by uv):
    uv run --with websockets --with httpx python harbor/relay/bridge.py
"""

import asyncio
import hashlib
import hmac
import json
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx
import websockets

RELAY_WS_URL = os.environ["RELAY_WS_URL"]
RELAY_SECRET = os.environ["RELAY_SECRET"]
VLLM_LOCAL_URL = os.environ.get("VLLM_LOCAL_URL", "http://localhost:8000").rstrip("/")
BACKEND_API_KEY = os.environ.get("BACKEND_API_KEY", "")  # injected into forwarded (sglang) calls
# Node A episode receiver: /episodes/* routes HERE (not the model). The receiver
# is a separate HTTP service (src/loop/episode_receiver) on the trainer node.
EPISODE_RECEIVER_URL = os.environ.get("EPISODE_RECEIVER_URL", "").rstrip("/")
CAPTURE_DIR = os.environ.get("CAPTURE_DIR", "")
WEIGHT_VERSION_FILE = os.environ.get("WEIGHT_VERSION_FILE", "")
MAX_GEN_TOKENS = int(os.environ.get("MAX_GEN_TOKENS", "0") or 0)
# Per-competition keys that sign `validate` scores (<dir>/<competition>/environment/score_key).
SCORE_KEYS_DIR = Path(os.environ.get("SCORE_KEYS_DIR", str(Path(__file__).resolve().parents[2] / "harbor_tasks" / "mlebench-low")))


def log(msg: str) -> None:
    print(f"[bridge] {msg}", flush=True, file=sys.stderr)


# --- Rollout capture (lifelong loop, node B) ----------------------------------
# Every /v1/chat/completions that passes through is recorded as one spool line:
# exact prompt/response token ids + per-token logprobs as sampled by SGLang, the
# served weight version, and validate boundaries. Agents address the relay as
# /sessions/<trial>/v1/..., so the trial id comes from the path; the prefix is
# stripped before the call reaches SGLang. Capture must never break proxying.
_SESSION_RE = re.compile(r"^/sessions/([^/]+)(/v1/.*)$")
# `validate <file.csv>` or `python validate.py <file.csv>`, matched in the shell command only
# (not the tool-call summary text, which mentions "validate" in prose).
_VALIDATE_RE = re.compile(r"(?:^|[\s;&|(`/])validate(?:\.py)?\s+\S+")
_SCORE_RE = re.compile(r"score=([-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf|None))(?:\s+sig=([0-9a-f]{16}))?")
_score_keys: dict[str, bytes | None] = {}
_ID_FIELDS = ("prompt_token_ids", "response_token_ids", "token_ids")
_write_lock = threading.Lock()
_turns: dict[str, int] = {}


def _score_key(competition: str) -> bytes | None:
    """Signing key for a competition, or None when its task has none (probe/test trials)."""
    if competition not in _score_keys:
        try:
            _score_keys[competition] = (SCORE_KEYS_DIR / competition / "environment" / "score_key").read_bytes().strip()
        except OSError:
            _score_keys[competition] = None
    return _score_keys[competition]


def _split_scores(trial: str, text: str) -> tuple[list[str], list[str], bool]:
    """(accepted, rejected, signed) scores in tool output. When the trial's competition has a
    key, only `score=<s> sig=<hmac>` lines signed by the root-only grader count; an agent
    printing `score=...` itself is rejected."""
    competition = trial.split("__")[0]
    key = _score_key(competition)
    accepted, rejected = [], []
    for score, sig in _SCORE_RE.findall(text):
        if key is None:
            accepted.append(score)
            continue
        want = hmac.new(key, f"{competition}:{score}".encode(), hashlib.sha256).hexdigest()[:16]
        (accepted if sig and hmac.compare_digest(sig, want) else rejected).append(score)
    return accepted, rejected, key is not None


def _weight_version() -> int | None:
    if not WEIGHT_VERSION_FILE:
        return None
    try:
        with open(WEIGHT_VERSION_FILE) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _text(content) -> str:
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return content or ""


def _command(tool_call: dict) -> str:
    """The shell command of a tool call (its JSON `command` argument), or ''."""
    try:
        args = json.loads((tool_call.get("function") or {}).get("arguments") or "{}")
    except Exception:
        return ""
    cmd = args.get("command") if isinstance(args, dict) else None
    return cmd if isinstance(cmd, str) else ""


def _prepare_chat(body: dict) -> dict:
    """Ask SGLang for token ids + logprobs; remember what the client asked for."""
    asked = {"logprobs": bool(body.get("logprobs")), "stream": bool(body.get("stream"))}
    if not asked["stream"]:
        body["logprobs"] = True
        body["return_prompt_token_ids"] = True
        body["return_token_ids"] = True
    return asked


def _cap_tokens(body: dict) -> bool:
    """Clamp the generation length to MAX_GEN_TOKENS; True if the cap was added to a request
    that set no limit. The agents set none, so a policy stuck in a repetition loop generated
    until the context filled (97k-token "WAIT WAIT ..." responses, 2026-11-04); a capped call
    ends with finish_reason "length"."""
    if MAX_GEN_TOKENS <= 0:
        return False
    limits = [k for k in ("max_tokens", "max_completion_tokens") if body.get(k) is not None]
    for k in limits:
        body[k] = min(int(body[k]), MAX_GEN_TOKENS)
    if not limits:
        body["max_tokens"] = MAX_GEN_TOKENS
    return not limits


def _strip_for_client(resp: dict, asked: dict) -> None:
    """Keep relay payloads small and client-visible responses unchanged."""
    for ch in resp.get("choices") or []:
        for k in _ID_FIELDS:
            ch.pop(k, None)
        if not asked["logprobs"]:
            ch.pop("logprobs", None)


def _build_record(trial: str, req: dict, resp: dict, v_start, v_end, latency_s: float) -> dict:
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    lp = (ch.get("logprobs") or {}).get("content") or []
    response_ids = ch.get("response_token_ids") or ch.get("token_ids")
    logprobs = [t.get("logprob") for t in lp]
    messages = req.get("messages") or []
    # New observations since the agent's last turn = trailing non-assistant messages.
    new_obs = []
    for m in reversed(messages):
        if m.get("role") == "assistant":
            break
        new_obs.append(m)
    new_obs.reverse()
    obs_text = "\n".join(_text(m.get("content")) for m in new_obs if m.get("role") == "tool")
    scores, rejected_scores, scores_signed = _split_scores(trial, obs_text)
    tool_calls = msg.get("tool_calls") or []
    calls_validate = any(_VALIDATE_RE.search(_command(tc)) for tc in tool_calls)
    system = next((_text(m.get("content")) for m in messages if m.get("role") == "system"), "")
    turn = _turns.get(trial, 0)
    _turns[trial] = turn + 1
    return {
        "id": uuid.uuid4().hex,
        "ts": time.time(),
        "latency_s": round(latency_s, 3),
        "trial": trial,
        "turn": turn,
        "weight_version": v_start,
        "weight_version_end": v_end,
        "prompt_token_ids": ch.get("prompt_token_ids"),
        "response_token_ids": response_ids,
        "response_logprobs": logprobs,
        "ids_logprobs_aligned": response_ids is not None and len(response_ids) == len(logprobs),
        "finish_reason": ch.get("finish_reason"),
        "sampling": {k: req.get(k) for k in ("temperature", "top_p", "top_k", "max_tokens", "max_completion_tokens")},
        "n_messages": len(messages),
        "system_hash": hashlib.sha1(system.encode()).hexdigest()[:12],
        "assistant_content": msg.get("content"),
        "tool_calls": tool_calls,
        "calls_validate": calls_validate,
        "observed_validate_scores": scores,
        "rejected_validate_scores": rejected_scores,
        "scores_signed": scores_signed,
        "usage": resp.get("usage"),
    }


def _append(trial: str, record: dict) -> None:
    safe = "".join(c for c in trial if c.isalnum() or c in "_-.") or "_untagged"
    path = os.path.join(CAPTURE_DIR, f"{safe}.jsonl")
    line = json.dumps(record) + "\n"
    with _write_lock:
        with open(path, "a") as f:
            f.write(line)


async def handle(ws, send_lock: asyncio.Lock, frame: dict, client: httpx.AsyncClient) -> None:
    """Proxy one relayed request to the local vLLM and return the response frame."""
    try:
        path = frame["path"]
        trial = "_untagged"
        m = _SESSION_RE.match(path)
        if m:
            trial, path = m.group(1), m.group(2)
        _is_episode = path.startswith("/episodes/")
        _base = EPISODE_RECEIVER_URL if _is_episode else VLLM_LOCAL_URL
        url = _base + path
        if frame.get("query"):
            url += "?" + frame["query"]
        _auth = ({"authorization": f"Bearer {BACKEND_API_KEY}"} if (BACKEND_API_KEY and not _is_episode) else {})
        content = frame["body"].encode()
        capture = None
        cap_added = False
        if (CAPTURE_DIR or MAX_GEN_TOKENS) and frame["method"] == "POST" and path.endswith("/chat/completions"):
            try:
                req = json.loads(frame["body"])
                cap_added = _cap_tokens(req)
                if CAPTURE_DIR:
                    asked = _prepare_chat(req)
                    if not asked["stream"]:
                        capture = (req, asked, _weight_version(), time.monotonic())
                content = json.dumps(req).encode()
            except Exception as e:
                log(f"capture prepare failed ({e}); forwarding unchanged")
        headers = {"content-type": "application/json", **_auth}
        r = await client.request(frame["method"], url, content=content, headers=headers)
        if cap_added and r.status_code == 400 and "context length" in r.text:
            # SGLang rejects prompt + max_tokens beyond the context; this close to the limit
            # the remaining room is already below the cap, so send the call without it.
            req.pop("max_tokens", None)
            content = json.dumps(req).encode()
            r = await client.request(frame["method"], url, content=content, headers=headers)
        body_text = r.text
        if capture is not None and r.status_code == 200:
            req, asked, v_start, t0 = capture
            try:
                resp = await asyncio.to_thread(json.loads, body_text)
                record = _build_record(trial, req, resp, v_start, _weight_version(), time.monotonic() - t0)
                await asyncio.to_thread(_append, trial, record)
                _strip_for_client(resp, asked)
                body_text = await asyncio.to_thread(json.dumps, resp)
            except Exception as e:
                log(f"capture failed for trial={trial} ({e}); response passed through")
        resp = {
            "id": frame["id"],
            "status": r.status_code,
            "body": body_text,
            "content_type": r.headers.get("content-type", "application/json"),
        }
    except Exception as e:  # return a 502 rather than dropping the request
        resp = {
            "id": frame["id"],
            "status": 502,
            "body": json.dumps({"error": {"message": f"bridge->vllm failed: {e}"}}),
            "content_type": "application/json",
        }
    # Serialize the vLLM response body (a completion, up to ~100s of KB) off the
    # event loop -- a synchronous json.dumps of a large payload here would block the
    # loop and starve the WebSocket keepalive (missed pongs -> the relay drops us
    # with a 1011). See main()'s ping settings.
    payload = await asyncio.to_thread(json.dumps, resp)
    # Serialize sends: concurrent handle() tasks writing the same socket would
    # interleave frames and trip a 1002 protocol error on the relay. If the socket
    # dropped while this request was in flight, log and drop -- the relay will time
    # the request out; a raw exception here is an unretrieved-task traceback.
    try:
        async with send_lock:
            await ws.send(payload)
    except Exception as e:
        log(f"dropping response {frame['id']}: ws send failed ({e})")


async def main() -> None:
    headers = {"Authorization": f"Bearer {RELAY_SECRET}"}
    if CAPTURE_DIR:
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        log(f"capturing chat completions -> {CAPTURE_DIR} (version file: {WEIGHT_VERSION_FILE or 'none'})")
    async with httpx.AsyncClient(timeout=float(os.environ.get("RELAY_REQUEST_TIMEOUT", "600"))) as client:
        while True:
            try:
                async with websockets.connect(
                    RELAY_WS_URL,
                    additional_headers=headers,
                    max_size=None,
                    # A momentarily-busy loop on either end must not be a death
                    # sentence: at 64-way concurrency the relay/bridge loops spend
                    # bursts shuttling large SWE-bench payloads, and a 20s pong
                    # deadline was tripping 1011 keepalive timeouts that dropped
                    # every in-flight request. Ping less often, wait far longer.
                    ping_interval=30,
                    ping_timeout=90,
                ) as ws:
                    log(f"connected to relay; forwarding to {VLLM_LOCAL_URL}")
                    send_lock = asyncio.Lock()  # one per connection
                    async for msg in ws:
                        # Parse the (large) request frame off the loop so keepalives
                        # keep flowing while a hefty prompt body deserializes.
                        frame = await asyncio.to_thread(json.loads, msg)
                        asyncio.create_task(handle(ws, send_lock, frame, client))
            except Exception as e:
                log(f"disconnected ({e}); reconnecting in 5s")
                await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
