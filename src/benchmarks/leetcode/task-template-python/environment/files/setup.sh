#!/usr/bin/env bash
set -euo pipefail
missing=0
echo '  -- python3 --version' && python3 --version || missing=$?
if [ "$missing" -ne 0 ]; then
    echo "[setup] python toolchain incomplete; aborting" >&2
    exit 1
fi
mkdir -p /opt/leetcode
cp /staging/env_files/adapters/* /opt/leetcode/
