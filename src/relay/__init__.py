"""Modal reverse-relay so a private node vLLM is reachable from Modal sandboxes.

Ported from tts-tokens-that-suffice. public_api.py = the public Modal app (HTTPS +
WebSocket node_forwarder endpoint); node_forwarder.py = the node-side leg that dials out and proxies to
local vLLM. Driven by scripts/eval/eval_terminal_bench_2.sh (RELAY=1)."""
