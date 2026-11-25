"""Local OpenAI-API proxy to a remote model server, for agents in a Babel job (Singularity shares the host network).

The agent talks to http://127.0.0.1:<port>/v1 with a dummy key; each request is forwarded to the upstream server with
the real key added here. So the key never appears on a command line (other users on a node can read those), in
Harbor's job config, or in the trial logs. Chat and text completions without a token limit get MAX_GEN_TOKENS, as
our local vLLM did (its default max_tokens; agents send no limit). Streaming responses are passed through as they
arrive.

    python -m src.relay.local_proxy --port 20123 --env-file ~/.config/msl/qwen9b.env --prefix QWEN9B

Env file (mode 600): <PREFIX>_BASE_URL (ending in /v1), <PREFIX>_API_KEY, and optionally <PREFIX>_MODEL.
Env: MAX_GEN_TOKENS (default 16384; 0 = no default limit).
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "host", "content-length", "authorization"}


def read_env(path: Path) -> dict[str, str]:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def make_handler(base: urllib.parse.SplitResult, key: str, max_gen: int):
    base_path = base.path.rstrip("/")  # e.g. /v1
    local = threading.local()

    def conn() -> http.client.HTTPConnection:
        c = getattr(local, "c", None)
        if c is None:
            cls = http.client.HTTPSConnection if base.scheme == "https" else http.client.HTTPConnection
            c = local.c = cls(base.netloc, timeout=3600)
        return c

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet: one line per error only
            pass

        def _forward(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if self.command == "POST" else None
            path = self.path
            if path.startswith("/v1"):
                path = path[3:]
            if body and max_gen and path.rstrip("/") in ("/chat/completions", "/completions"):
                try:
                    req = json.loads(body)
                    if not req.get("max_tokens") and not req.get("max_completion_tokens"):
                        req["max_tokens"] = max_gen
                        body = json.dumps(req).encode()
                except ValueError:
                    pass
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            headers["Authorization"] = f"Bearer {key}"
            if body is not None:
                headers["Content-Length"] = str(len(body))
            for attempt in (0, 1):  # a kept-alive upstream connection may have been closed: reconnect once
                c = conn()
                try:
                    c.request(self.command, base_path + path, body=body, headers=headers)
                    r = c.getresponse()
                    break
                except (http.client.HTTPException, OSError):
                    c.close()
                    local.c = None
                    if attempt:
                        self.send_error(502, "upstream unreachable")
                        return
            self.send_response(r.status)
            for k, v in r.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                while True:
                    chunk = r.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):  # the agent hung up: drop the upstream connection too
                c.close()
                local.c = None
            if r.will_close:
                c.close()
                local.c = None

        do_GET = do_POST = _forward

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--env-file", required=True)
    ap.add_argument("--prefix", required=True, help="variable prefix in the env file, e.g. QWEN9B")
    a = ap.parse_args()
    env = read_env(Path(a.env_file).expanduser())
    base, key = env.get(f"{a.prefix}_BASE_URL"), env.get(f"{a.prefix}_API_KEY")
    if not base or not key:
        sys.exit(f"{a.env_file}: needs {a.prefix}_BASE_URL and {a.prefix}_API_KEY")
    max_gen = int(os.environ.get("MAX_GEN_TOKENS", "16384"))
    srv = ThreadingHTTPServer(("127.0.0.1", a.port),
                              make_handler(urllib.parse.urlsplit(base), key, max_gen))
    srv.daemon_threads = True
    print(f"local_proxy: 127.0.0.1:{a.port} -> {base} (max_tokens default {max_gen})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
