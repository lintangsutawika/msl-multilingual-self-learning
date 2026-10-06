#!/usr/bin/env bash
set -euo pipefail

echo '  -- python3 --version'
python3 --version

echo '  -- node --version'
node --version

echo '  -- tsc --version'
tsc --version

mkdir -p /opt/leetcode
cp /staging/env_files/adapters/* /opt/leetcode/
mkdir -p /workspace
[ -d /solution ] && cp -r /solution/. /workspace/ 2>/dev/null || true


echo "[setup] TypeScript environment ready"
