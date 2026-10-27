#!/usr/bin/env bash
# Run at Sailbox/container start (Dockerfile CMD). Prepares the competition data
# once so the agent + validate server never need to trigger `mlebench prepare`
# at runtime, then starts the validate server and keeps the box alive.
set -euo pipefail
COMPETITION="{comp_id}"

# 1. Neutralize Kaggle's interactive rule-acceptance prompts (headless -> EOF).
python3 /workspace/neutralize_rules.py

# 2. Prepare the competition data (Kaggle creds + accepts rules), then expose the
#    agent-facing splits at a clean, stable path: /workspace/public/.
echo "[startup] preparing $COMPETITION ..."
mlebench prepare -c "$COMPETITION"
echo "[startup] prepared $COMPETITION"
echo "[startup] exposing prepared/public at /workspace/public ..."
rm -rf /workspace/public && mkdir -p /workspace/public
python3 - <<PY
from mlebench.registry import registry
import shutil, pathlib
comp = registry.get_competition("$COMPETITION")
src = pathlib.Path(comp.public_dir)
dst = pathlib.Path("/workspace/public")
shutil.copytree(src, dst, dirs_exist_ok=True)
print("copied", src, "->", dst)
PY
echo "[startup] public data at /workspace/public: $(ls /workspace/public)"

# 3. Start the validate server in the background.
python3 /workspace/validate_server.py &
echo "[startup] validate server started on :5000"

# 4. Keep the Sailbox alive for agent execs.
exec tail -f /dev/null
