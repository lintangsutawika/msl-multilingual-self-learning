#!/bin/bash
# MLE-Bench baseline of serving-only eval endpoints: runs src/rollout/dispatch.py (the
# training rollouts' dispatcher: OpenHands on SAIL, same agent settings) against each
# endpoint's run dir (endpoint.env), CPU only. Trials and grades land in
# <run dir>/trials; summarize with the verifier grade.json files.
#
#   env EVAL_RUNS=eval-qwen9b:qwen3.5-9B,eval-qwen36-27b:qwen3.6-27B ROUNDS=3 CONCURRENT=8 \
#     qsub -N mle-baseline -j oe -o logs/ -P gai51740 -q R9920261300 -V -v RTYPE=rt_HC \
#       -l select=1 -l walltime=48:00:00 scripts/eval/mlebench_baseline.sh
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${PBS_O_WORKDIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$REPO_ROOT"
MSL_STORE="${MSL_STORE:-/groups/gai51740/aci18914wh/msl/runs}"
TASKS="${TASKS:-denoising-dirty-documents,leaf-classification,nomad2018-predict-transparent-conductors,plant-pathology-2020-fgvc7,random-acts-of-pizza,spooky-author-identification,tabular-playground-series-dec-2021,tabular-playground-series-may-2022}"
JOBTAG="${PBS_JOBID:-local-$$}"; JOBTAG="${JOBTAG%%.*}"
PIDS=()
trap 'for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null || true; done; wait' EXIT TERM INT
for pair in ${EVAL_RUNS//,/ }; do
  run="${pair%%:*}"; model="${pair#*:}"
  .venv/bin/python src/rollout/dispatch.py --run-dir "${MSL_STORE}/${run}" --tasks "$TASKS" --model "$model" \
    --rounds "${ROUNDS:-3}" --max-concurrent "${CONCURRENT:-8}" --max-iterations "${MAX_ITERATIONS:-200}" \
    --condenser-max-events "${CONDENSER_MAX_EVENTS:-120}" --terminal-max-timeout "${TERMINAL_MAX_TIMEOUT:-1800}" \
    > "logs/${JOBTAG}-baseline-${run}.log" 2>&1 &
  PIDS+=($!)
  echo "[baseline] ${run} (${model}) -> logs/${JOBTAG}-baseline-${run}.log"
done
wait
