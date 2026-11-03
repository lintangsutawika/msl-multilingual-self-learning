#!/usr/bin/env bash
# Run at Sailbox/container start (Dockerfile CMD). Data is BAKED into the image
# (/workspace/{private,public,leaderboard.csv}); place it where mlebench grade reads
# them -- NO runtime Kaggle download/prepare -- then start the validate server.
set -euo pipefail
COMPETITION="{comp_id}"

# 1. Neutralize Kaggle's interactive rule-acceptance prompts (harmless no-op now).
python3 /opt/msl/neutralize_rules.py

# 2. Place baked data at the mlebench registry paths + expose /workspace/public.
echo "[startup] placing baked data for $COMPETITION (no Kaggle)"
python3 - <<'PY'
from mlebench.registry import registry
import shutil, pathlib
comp = registry.get_competition("$COMPETITION")
pub = comp.public_dir
priv = comp.private_dir
lb = comp.leaderboard
lb.parent.mkdir(parents=True, exist_ok=True)
if pathlib.Path('/workspace/private').exists():
    priv.mkdir(parents=True, exist_ok=True)
    shutil.copytree('/workspace/private', priv, dirs_exist_ok=True)
if pathlib.Path('/workspace/public').exists():
    pub.mkdir(parents=True, exist_ok=True)
    shutil.copytree('/workspace/public', pub, dirs_exist_ok=True)
if pathlib.Path('/workspace/leaderboard.csv').exists():
    shutil.copy('/workspace/leaderboard.csv', lb)
print('[startup] baked private/public/leaderboard placed (no Kaggle)')
PY
echo "[startup] public data at /workspace/public: $(ls /workspace/public 2>/dev/null)"

# 3. Start the validate server in the background.
python3 /opt/msl/validate_server.py &
echo "[startup] validate server started on :5000"

# 4. Keep the Sailbox alive for agent execs.
exec tail -f /dev/null
