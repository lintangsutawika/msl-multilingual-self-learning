"""Host-side `validate` grader for MLE-bench tasks run on Babel (run_babel.sh starts one per task run).

Rootless Singularity on Babel has no real uid separation (no /etc/subuid entry, so `su agent` only fakes the uid):
anything inside the container is readable by the agent. So the validation answers and the score-signing key never
enter the container. This server runs on the host, outside it, reads them from the task's data dir, and grades
submissions sent by the container's `validate` command over 127.0.0.1 (Singularity shares the host network).

    python validate_host.py --competition ID --answers DIR --key FILE --endpoint-dir DIR --log FILE

Writes DIR/endpoint ("<url> <token>"); run_babel.sh bind-mounts DIR read-only at /etc/msl-validate. Requests must
carry the token (other users' jobs on the node can reach the port, but not read that file). Run it with the grader
environment (src/benchmarks/mlebench/eval: the image's MLE-bench revision and pandas<3, so scores match).
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # grade_subset.py, next to this file
from grade_subset import grade  # noqa: E402

MAX_BYTES = 4 * 2**30


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--competition", required=True)
    ap.add_argument("--answers", required=True, help="validation answers dir (valsplit/private_val)")
    ap.add_argument("--key", required=True, help="the task's score-signing key (environment/score_key)")
    ap.add_argument("--endpoint-dir", required=True)
    ap.add_argument("--log", required=True, help="every graded request, appended as jsonl")
    a = ap.parse_args()
    comp, answers = a.competition, a.answers
    key = Path(a.key).read_bytes().strip()
    token = secrets.token_hex(16)
    lock = threading.Lock()  # one grade at a time: graders can be memory-hungry

    def log(rec: dict) -> None:
        with open(a.log, "a") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}) + "\n")

    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            if not hmac.compare_digest(self.headers.get("X-Token", ""), token):
                return self._send(403, {"error": "bad token"})
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= MAX_BYTES:
                return self._send(400, {"error": f"submission must be 1 byte to {MAX_BYTES >> 30} GiB"})
            name = self.headers.get("X-Name", "")[:200]
            with tempfile.TemporaryDirectory() as work:
                sub = Path(work) / "submission.csv"
                with open(sub, "wb") as f:
                    left = n
                    while left:
                        chunk = self.rfile.read(min(left, 1 << 20))
                        if not chunk:
                            return self._send(400, {"error": "truncated upload"})
                        f.write(chunk)
                        left -= len(chunk)
                try:
                    with lock:
                        score = grade(str(sub), comp, answers)
                except Exception as e:  # invalid submission (ValueError) or a grader crash: tell the agent why
                    log({"name": name, "error": str(e)[:500]})
                    return self._send(422, {"error": str(e)[:2000]})
            text = f"{score}"
            sig = hmac.new(key, f"{comp}:{text}".encode(), hashlib.sha256).hexdigest()[:16]
            log({"name": name, "score": score})
            self._send(200, {"score": text, "sig": sig})

        def log_message(self, *args) -> None:
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    ep = Path(a.endpoint_dir)
    ep.mkdir(parents=True, exist_ok=True)
    tmp = ep / ".endpoint.tmp"
    tmp.write_text(f"http://127.0.0.1:{srv.server_address[1]}/ {token}\n")
    os.replace(tmp, ep / "endpoint")
    print(f"validate_host: {comp} on 127.0.0.1:{srv.server_address[1]}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
