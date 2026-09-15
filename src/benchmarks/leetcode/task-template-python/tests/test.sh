#!/usr/bin/env bash
set -uo pipefail
# Submit hook: package agent source -> solution.json.
if [ -f /tests/_package_submission.py ]; then python3 /tests/_package_submission.py || true; fi
python3 /tests/test.py
