#!/bin/bash
# Two serving-only eval endpoints on one 8-GPU node (rt_HF): GPUs 0-3 and 4-7, each a full
# scripts/rollout.sh (SGLang DP4 + its own Modal relay), with its own run dir and log dir.
# ABCI has no 4-GPU resource type, so this is how two models get 4 GPUs each.
#
#   env MODEL_A=Qwen/Qwen3.6-27B NAME_A=eval-qwen36-27b PORT_A=30013 \
#       MODEL_B=Qwen/Qwen3.8-27B NAME_B=eval-qwen38-27b PORT_B=30014 \
#     qsub -N eval-27b -j oe -o logs/ -P gai51740 -q R9920261300 -V -v RTYPE=rt_HF \
#       -l select=1 -l walltime=168:00:00 scripts/serve/eval_pair.sh
#
# Models are downloaded into the HF cache first (skipped if already there).
# MODEL_A/B may also be local checkpoint dirs. Other scripts/rollout.sh settings pass through the
# environment, e.g. DISPATCH_TASKS / DISPATCH_TASKS_DIR / DISPATCH_ROUNDS to run trials against
# each endpoint, EPISODE_BUDGET, MAX_GEN_TOKENS (default 0 = uncapped).
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${PBS_O_WORKDIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$REPO_ROOT"
JOBTAG="${PBS_JOBID:-local-$$}"; JOBTAG="${JOBTAG%%.*}"
log() { echo "[eval-pair $(date +%H:%M:%S)] $*"; }

fetch() {  # repo id (or a local checkpoint dir, used as is) -> local snapshot path
  if [ -d "$1" ]; then echo "$1"; return; fi
  # A local path that does not exist is an error, not a repo id: the empty path made rollout.sh fall back to
  # the base model, so evals of a crashed run silently scored base (TIS seeds 2-4, 2026-11-16).
  case "$1" in /*) echo "checkpoint $1 does not exist" >&2; return 1;; esac
  uv run -q --no-project --with huggingface_hub python -c \
    "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))" "$1" | tail -1
}
log "downloading ${MODEL_A} and ${MODEL_B}"
fetch "$MODEL_A" > "/tmp/eval-pair-${JOBTAG}-a" &
PA=$!
fetch "$MODEL_B" > "/tmp/eval-pair-${JOBTAG}-b" &
PB=$!
wait "$PA" && wait "$PB"
PATH_A="$(cat "/tmp/eval-pair-${JOBTAG}-a")"; PATH_B="$(cat "/tmp/eval-pair-${JOBTAG}-b")"
log "A: ${PATH_A}"; log "B: ${PATH_B}"

PIDS=()
trap 'for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null || true; done; wait' EXIT TERM INT
serve() {  # name port gpus model_path [tasks]  (tasks: DISPATCH_TASKS_A/_B override DISPATCH_TASKS)
  env RUN_NAME="$1" PORT="$2" CUDA_VISIBLE_DEVICES="$3" MODEL_PATH="$4" RELAY_APP="msl-$1" \
    DISPATCH_TASKS="${5:-${DISPATCH_TASKS:-}}" \
    LOG_DIR="${REPO_ROOT}/logs/rollout-${JOBTAG}-$1" CAPTURE=0 DP_SIZE=4 TP_SIZE=1 MAX_GEN_TOKENS="${MAX_GEN_TOKENS:-0}" \
    bash scripts/rollout.sh > "${REPO_ROOT}/logs/${JOBTAG}-$1.out" 2>&1 &
  PIDS+=($!)
  log "$1 on GPUs $3, port $2 (log logs/${JOBTAG}-$1.out)"
}
serve "$NAME_A" "$PORT_A" 0,1,2,3 "$PATH_A" "${DISPATCH_TASKS_A:-}"
serve "$NAME_B" "$PORT_B" 4,5,6,7 "$PATH_B" "${DISPATCH_TASKS_B:-}"
# Wait for both endpoints: each ends on its own once its trials finish (DISPATCH_ROUNDS > 0),
# so the faster model no longer cuts the slower one off.
for p in "${PIDS[@]}"; do wait "$p" || log "endpoint pid $p exited with $?"; done
log "both endpoints finished"
