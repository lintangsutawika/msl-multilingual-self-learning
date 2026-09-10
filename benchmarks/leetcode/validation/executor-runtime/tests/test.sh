#!/usr/bin/env bash

set -uo pipefail

mkdir -p /logs/verifier

{
    apt-get update &&
    apt-get install -y \
        g++ \
        nlohmann-json3-dev \
        golang-go \
        openjdk-17-jdk-headless \
        libgoogle-gson-java
} > /logs/verifier/setup.txt 2>&1

SETUP_STATUS=$?

if [ "$SETUP_STATUS" -ne 0 ]; then
    echo "RUNTIME_ERROR" > /logs/verifier/status.txt
    echo "0" > /logs/verifier/reward.txt
    exit 0
fi

export PYTHONPATH="/tests:${PYTHONPATH:-}"

python3 -m \
    msl_multilingual_self_learning.benchmark.leetcode.audit_executor_runtime \
    > /logs/verifier/test-stdout.txt \
    2>&1

STATUS=$?

if [ "$STATUS" -eq 0 ]; then
    echo "PASS" > /logs/verifier/status.txt
    echo "1" > /logs/verifier/reward.txt
else
    echo "RUNTIME_ERROR" > /logs/verifier/status.txt
    echo "0" > /logs/verifier/reward.txt
fi

exit 0