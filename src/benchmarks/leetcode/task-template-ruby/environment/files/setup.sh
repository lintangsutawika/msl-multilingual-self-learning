#!/usr/bin/env bash
set -euo pipefail

python3 --version
ruby --version

mkdir -p /opt/leetcode
cp /staging/env_files/adapters/* /opt/leetcode/
mkdir -p /workspace
[ -d /solution ] && cp -r /solution/. /workspace/ 2>/dev/null || true


echo "[setup] Ruby environment ready"
