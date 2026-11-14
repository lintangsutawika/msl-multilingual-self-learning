#!/usr/bin/env bash
# Oracle check (Harbor's `oracle` agent): submit the competition's sample submission, so the plumbing is tested end to
# end (data mount, `validate` on the validation rows, final grade on MLE-bench's test rows). Not a real solution: the
# score is the sample's. The agent never sees this file.
set -uo pipefail
cd /workspace
SAMPLE=$(find /workspace/public -maxdepth 1 -type f \( -iname '*sample*submission*.csv' -o -iname '*sample*submission*.csv.zip' \) | sort | head -1)
[ -n "$SAMPLE" ] || { echo "oracle: no sample submission in /workspace/public" >&2; exit 1; }
case "$SAMPLE" in
  *.zip) python3 -c 'import sys, zipfile; z = zipfile.ZipFile(sys.argv[1]); open("/workspace/predictions.csv", "wb").write(z.read(z.namelist()[0]))' "$SAMPLE" ;;
  *) cp "$SAMPLE" /workspace/predictions.csv ;;
esac
echo "oracle: submitted $(basename "$SAMPLE") ($(wc -l < /workspace/predictions.csv) lines)"
validate /workspace/predictions.csv || echo "oracle: validate failed (rc=$?)"
