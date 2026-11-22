#!/usr/bin/env bash
# Oracle smoke test: submit the sample submission and call validate once.
cp '/workspace/public/{sample_submission}' /workspace/predictions.csv
validate /workspace/predictions.csv; echo "validate rc=$?"
