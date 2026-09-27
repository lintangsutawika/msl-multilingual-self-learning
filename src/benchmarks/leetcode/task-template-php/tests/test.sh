#!/usr/bin/env bash
set -uo pipefail
# Verifier: test.py reads the agent solution (packaging folded in) and grades it.
python3 /tests/test.py
