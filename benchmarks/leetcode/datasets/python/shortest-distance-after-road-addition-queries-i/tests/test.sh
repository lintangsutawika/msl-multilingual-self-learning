#!/usr/bin/env bash

set -uo pipefail

# Harbor expects verifier outputs here.
mkdir -p /logs/verifier

# Run the Python verifier.
python /tests/test.py
STATUS=$?

# Translate the verifier's exit code into a Harbor reward.
case "$STATUS" in
    0)
        echo "1" > /logs/verifier/reward.txt
        echo "PASS" > /logs/verifier/status.txt
        ;;

    1)
        echo "0" > /logs/verifier/reward.txt
        echo "WRONG_ANSWER" > /logs/verifier/status.txt
        ;;

    *)
        echo "0" > /logs/verifier/reward.txt
        echo "RUNTIME_ERROR" > /logs/verifier/status.txt
        ;;
esac

# The verifier itself completed successfully.
# A wrong model answer is represented by reward=0, not by crashing test.sh.
exit 0