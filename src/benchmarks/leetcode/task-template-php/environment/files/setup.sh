#!/usr/bin/env bash
set -euo pipefail

python3 --version
php --version

mkdir -p /opt/leetcode
cp /staging/env_files/adapters/* /opt/leetcode/

echo "[setup] PHP environment ready"
