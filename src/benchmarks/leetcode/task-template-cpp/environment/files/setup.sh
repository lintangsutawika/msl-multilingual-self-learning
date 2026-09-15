#!/usr/bin/env bash
set -euo pipefail
missing=0
echo '  -- python3 --version' && python3 --version || missing=$?
echo '  -- g++ --version | head -n 1' && g++ --version | head -n 1 || missing=$?
echo '  -- [ -f /usr/include/nlohmann/json.hpp ]' && [ -f /usr/include/nlohmann/json.hpp ] || missing=$?
if [ "$missing" -ne 0 ]; then
    echo "[setup] cpp toolchain incomplete; aborting" >&2
    exit 1
fi
mkdir -p /opt/leetcode
cp /staging/env_files/adapters/* /opt/leetcode/
