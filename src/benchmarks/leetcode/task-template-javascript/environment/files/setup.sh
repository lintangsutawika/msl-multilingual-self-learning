#!/usr/bin/env bash
set -euo pipefail

echo '  -- python3 --version'
python3 --version

echo '  -- node --version'
node --version

mkdir -p /opt/leetcode
cp /staging/env_files/adapters/* /opt/leetcode/

echo "[setup] JavaScript environment ready"
