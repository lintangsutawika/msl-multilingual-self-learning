"""Rollout weight watcher: hot-swap SGLang to the checkpoint the trainer published.

The trainer writes an HF export (hf/step_N) and then atomically replaces a LATEST file:

    {"version": N, "path": "/groups/.../<trainer run>/hf/step_N"}

This process (inside the rollout job, so it can reach SGLang on localhost with the
admin key) polls LATEST. When the version is newer than what is served it calls
SGLang's /update_weights_from_disk (in-flight requests finish first; the prefix
cache is flushed), then writes the new version to WEIGHT_VERSION_FILE, which the
forwarder stamps on every spool record.

Self-check: before and after each swap it scores a fixed text with /generate
(prompt logprobs) and logs the largest per-token change, so a swap that only
relabels the version without changing the weights is visible.

Env:
    LATEST_FILE          the trainer's LATEST pointer to watch
    SGLANG_URL           e.g. http://127.0.0.1:30010
    API_KEY_FILE / ADMIN_KEY_FILE   SGLang keys written by scripts/rollout.sh
    WEIGHT_VERSION_FILE  served version (int), read by the forwarder
    POLL_SECS            default 30
"""
import json
import os
import sys
import time
from pathlib import Path

import httpx

LATEST_FILE = Path(os.environ["LATEST_FILE"])
SGLANG_URL = os.environ.get("SGLANG_URL", "http://127.0.0.1:30010").rstrip("/")
API_KEY = Path(os.environ["API_KEY_FILE"]).read_text().strip()
ADMIN_KEY = Path(os.environ["ADMIN_KEY_FILE"]).read_text().strip()
VERSION_FILE = Path(os.environ["WEIGHT_VERSION_FILE"])
POLL_SECS = float(os.environ.get("POLL_SECS", "30"))
CHECK_TEXT = ("The quick brown fox jumps over the lazy dog. In machine learning, a model is trained "
              "on data and evaluated on a held-out split to estimate how well it generalizes.")


def log(msg: str) -> None:
    print(f"[weight_watcher {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def served_version() -> int:
    try:
        return int(VERSION_FILE.read_text().strip())
    except Exception:
        return 0


def read_latest() -> dict | None:
    try:
        d = json.loads(LATEST_FILE.read_text())
        return {"version": int(d["version"]), "path": str(d["path"])}
    except Exception:
        return None


def prompt_logprobs(client: httpx.Client) -> list[float] | None:
    try:
        r = client.post(f"{SGLANG_URL}/generate", headers={"Authorization": f"Bearer {API_KEY}"}, json={
            "text": CHECK_TEXT, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            "return_logprob": True, "logprob_start_len": 0})
        r.raise_for_status()
        return [x[0] for x in r.json()["meta_info"]["input_token_logprobs"] if x[0] is not None]
    except Exception as e:
        log(f"self-check scoring failed: {e}")
        return None


def swap(client: httpx.Client, latest: dict) -> bool:
    before = prompt_logprobs(client)
    t0 = time.time()
    r = client.post(f"{SGLANG_URL}/update_weights_from_disk",
                    headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                    json={"model_path": latest["path"], "weight_version": str(latest["version"]),
                          "flush_cache": True, "abort_all_requests": False},
                    timeout=1800)
    try:
        ok = r.status_code == 200 and r.json().get("success", True)
    except ValueError:
        ok = False
    if not ok:
        log(f"swap to v{latest['version']} FAILED ({r.status_code}): {r.text[:300]}")
        return False
    tmp = VERSION_FILE.with_suffix(".tmp")
    tmp.write_text(str(latest["version"]))
    os.replace(tmp, VERSION_FILE)
    after = prompt_logprobs(client)
    delta = (max(abs(a - b) for a, b in zip(before, after))
             if before and after and len(before) == len(after) else None)
    info = client.get(f"{SGLANG_URL}/model_info", headers={"Authorization": f"Bearer {API_KEY}"}).json()
    log(f"swapped to v{latest['version']} from {latest['path']} in {time.time() - t0:.1f}s; "
        f"server weight_version={info.get('weight_version')}; "
        f"self-check max |Δ logprob| over {len(after or [])} tokens = {delta}")
    return True


def main() -> int:
    log(f"watching {LATEST_FILE}; serving v{served_version()}")
    with httpx.Client(timeout=120) as client:
        while True:
            latest = read_latest()
            if latest and latest["version"] > served_version():
                if not Path(latest["path"]).is_dir():
                    log(f"LATEST points at missing {latest['path']}; waiting")
                else:
                    swap(client, latest)
            time.sleep(POLL_SECS)


if __name__ == "__main__":
    sys.exit(main())
