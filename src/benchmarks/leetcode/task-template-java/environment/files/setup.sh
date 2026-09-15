#!/usr/bin/env bash
set -euo pipefail
missing=0
echo '  -- python3 --version' && python3 --version || missing=$?
echo '  -- javac -version' && javac -version || missing=$?
echo '  -- [ -f /usr/share/java/gson.jar ]' && [ -f /usr/share/java/gson.jar ] || missing=$?
if [ "$missing" -ne 0 ]; then
    echo "[setup] java toolchain incomplete; aborting" >&2
    exit 1
fi
mkdir -p /opt/leetcode
cp /staging/env_files/adapters/* /opt/leetcode/
