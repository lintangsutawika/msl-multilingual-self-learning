#!/usr/bin/env bash
# Rollout job of the lifelong loop: SGLang + per-job Modal relay + capturing forwarder,
# all inside ONE qsub job (separate jobs cannot reach each other's ports).
#
#   agents (SAIL) --https--> relay /sessions/<trial>/v1 --ws--> node_forwarder --> SGLang :PORT
#                                                                   |
#                                                                   +--> runs/<RUN_NAME>/spool/<trial>.jsonl
#
# The trainer never talks to this job directly: it reads the spool and (later) writes
# checkpoints + LATEST on the shared FS, which the weight watcher here applies.
#
# Submit:
#   qsub -N rollout -j oe -o logs/ -P gai51740 -q R9920261300 -v RTYPE=rt_HF \
#        -l select=1 -l walltime=120:00:00 -V scripts/rollout.sh
# Env:
#   MILES_SIF    miles.sif (MILES sglang fork; default: the shared one)
#   MODEL_PATH   HF checkpoint to serve (default: base Qwen3.5-9B snapshot)
#   MSL_STORE    group-storage root (default /groups/gai51740/aci18914wh/msl/runs)
#   RUN_NAME     run dir under MSL_STORE for keys, endpoint, spool, version (default rollout);
#                must differ per concurrent job (the script refuses a run dir a live job uses)
#   CAPTURE      1 = record every call to the spool (default); 0 = serving-only endpoint
#   RELAY_APP    Modal app name (default msl-rollout-relay-<job>; a fixed name keeps the URL)
#   PORT         local SGLang port (default 30010)
#   TP_SIZE / DP_SIZE / CONTEXT_LENGTH   SGLang parallelism / context (default 1 / 1 / 262144)
#   SGLANG_EXTRA_ARGS  extra SGLang flags (default: --tool-call-parser qwen3_coder)
#   LATEST_FILE        the trainer run's LATEST pointer to hot-swap from (default: none)
#   DISPATCH_TASKS     comma-separated mlebench-low tasks to run as real trials (default: none)
#   DISPATCH_MAX_CONCURRENT / MAX_ITERATIONS   dispatcher limits (default 2 / 200)
#   DISPATCH_ROUNDS    passes over DISPATCH_TASKS (default 1; 0 = forever)
#   CONDENSER_MAX_EVENTS  agent summarizes older history past this many events (default 120; 0 = off)
#   MAX_GEN_TOKENS     cap on tokens per model call, set by the forwarder (default 16384; 0 = no cap)
#   TERMINAL_MAX_TIMEOUT  cap on how long one agent terminal action may block, seconds (default 1800; 0 = no cap)
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${PBS_O_WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT"

JOBTAG="${PBS_JOBID:-local-$$}"; JOBTAG="${JOBTAG%%.*}"
RUN_NAME="${RUN_NAME:-rollout}"
# All the rollout job state (keys, endpoint, weight version, spool) on group storage: the spool keeps
# full prompt token ids per call and grows fast; /home has a 2 TiB quota. Secrets are umask 077.
MSL_STORE="${MSL_STORE:-/groups/gai51740/aci18914wh/msl/runs}"
RUN_DIR="${MSL_STORE}/${RUN_NAME}"
SPOOL_DIR="${SPOOL_DIR:-${RUN_DIR}/spool}"
# CAPTURE=0: serving-only endpoint (e.g. for evaluation): nothing is recorded to the spool.
CAPTURE="${CAPTURE:-1}"
if [ "$CAPTURE" = 1 ]; then FWD_CAPTURE_DIR="$SPOOL_DIR"; else FWD_CAPTURE_DIR=""; fi
LOG_DIR="${REPO_ROOT}/logs/rollout-${JOBTAG}"
PORT="${PORT:-30010}"
MILES_SIF="${MILES_SIF:-/home/aci18914wh/mho-model-harness-optimization/miles.sif}"
MODEL_PATH="${MODEL_PATH:-/home/aci18914wh/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
RELAY_APP="${RELAY_APP:-msl-rollout-relay-${JOBTAG}}"
MAX_GEN_TOKENS="${MAX_GEN_TOKENS:-16384}"
mkdir -p "$RUN_DIR" "$SPOOL_DIR" "$LOG_DIR"
log() { echo "[rollout $(date +%H:%M:%S)] $*"; }
log "job=${JOBTAG} run_dir=${RUN_DIR} model=${MODEL_PATH} port=${PORT}"

# Refuse to take over a run dir another live job is serving from (keys, endpoint.env and
# weight_version would be overwritten, and its cleanup would delete our endpoint.env).
if [ -f "${RUN_DIR}/endpoint.env" ]; then
  OTHER="$(sed -n 's/^ROLLOUT_JOB=//p' "${RUN_DIR}/endpoint.env")"
  if [ -n "$OTHER" ] && [ "$OTHER" != "$JOBTAG" ] && qstat "${OTHER}.pbs1" >/dev/null 2>&1; then
    log "RUN_DIR ${RUN_DIR} is in use by running job ${OTHER}; set a different RUN_NAME"
    exit 1
  fi
fi

# Served weight version: SGLang always starts from MODEL_PATH (= version 0); the weight
# watcher bumps it after each swap. Reset on every start so a restart never reports a stale version.
echo 0 > "${RUN_DIR}/weight_version"

PIDS=()
cleanup() {
  log "shutting down (stopping relay app ${RELAY_APP})"
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
  # -y: without a terminal `modal app stop` asks for confirmation and aborts, which
  # left the app (and its pinned container) running.
  uv run modal app stop -y "${RELAY_APP}" >/dev/null 2>&1 || log "could not stop relay app ${RELAY_APP}; stop it by hand"
  rm -f "${RUN_DIR}/endpoint.env"
}
trap cleanup EXIT TERM INT

# 1. SGLang (MILES fork: has /update_weights_from_disk), bound to localhost only.
# Fresh per-launch keys: api_key gates /v1 (only the forwarder holds it), admin key
# gates /update_weights_from_disk (only the weight watcher in this job reads it).
API_KEY="$(openssl rand -hex 32)"
ADMIN_API_KEY="$(openssl rand -hex 32)"
umask 077
printf '%s' "$API_KEY" > "${RUN_DIR}/sglang_api_key"
printf '%s' "$ADMIN_API_KEY" > "${RUN_DIR}/sglang_admin_api_key"
umask 022
# Qwen3.5 agents need structured tool calls: qwen3_coder parser, and NO reasoning
# parser (it swallows <tool_call> emitted before an omitted </think>; see configs/serve/Qwen).
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:---tool-call-parser qwen3_coder}"
singularity exec --nv --writable-tmpfs --bind /home:/home --bind "${MSL_STORE}:${MSL_STORE}" "$MILES_SIF" \
  python3 /sgl-workspace/sglang/python/sglang/launch_server.py \
  --model-path "$MODEL_PATH" --host 127.0.0.1 --port "$PORT" \
  --tp-size "${TP_SIZE:-1}" --dp-size "${DP_SIZE:-1}" --context-length "${CONTEXT_LENGTH:-262144}" \
  --api-key "$API_KEY" --admin-api-key "$ADMIN_API_KEY" \
  $SGLANG_EXTRA_ARGS > "${LOG_DIR}/sglang.log" 2>&1 &
SGLANG_PID=$!; PIDS+=("$SGLANG_PID")
log "sglang pid=${SGLANG_PID}; waiting for health"
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
  kill -0 "$SGLANG_PID" 2>/dev/null || { log "sglang exited during startup"; tail -30 "${LOG_DIR}/sglang.log"; exit 1; }
  sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || { log "sglang not healthy after 30 min"; exit 1; }
log "sglang healthy"

# 2. Per-job Modal relay (public /v1 + /sessions/<trial>/v1 only; no admin routes).
RELAY_SECRET="$(openssl rand -hex 16)"
DEPLOY_OUT="$(RELAY_APP_NAME="$RELAY_APP" RELAY_SECRET_NAME="$RELAY_APP" RELAY_SECRET="$RELAY_SECRET" \
  uv run modal deploy src/relay/public_api.py 2>&1)" || { echo "$DEPLOY_OUT"; log "relay deploy failed"; exit 1; }
RELAY_URL="$(printf '%s' "$DEPLOY_OUT" | grep -oE 'https://[a-zA-Z0-9._-]+\.modal\.run' | head -1)"
[ -n "$RELAY_URL" ] || { echo "$DEPLOY_OUT"; log "could not parse relay URL"; exit 1; }
log "relay deployed: ${RELAY_URL}"

# 3. Forwarder: relay bridge + per-call capture into the shared spool.
RELAY_WS_URL="wss://${RELAY_URL#https://}/bridge" RELAY_SECRET="$RELAY_SECRET" \
  VLLM_LOCAL_URL="http://127.0.0.1:${PORT}" BACKEND_API_KEY="$API_KEY" \
  CAPTURE_DIR="$FWD_CAPTURE_DIR" WEIGHT_VERSION_FILE="${RUN_DIR}/weight_version" MAX_GEN_TOKENS="$MAX_GEN_TOKENS" \
  uv run --with websockets --with httpx python src/relay/node_forwarder.py > "${LOG_DIR}/forwarder.log" 2>&1 &
FWD_PID=$!; PIDS+=("$FWD_PID")

# Endpoint for dispatchers/agents (secret: owner-only).
umask 077
cat > "${RUN_DIR}/endpoint.env" <<EOF
RELAY_URL=${RELAY_URL}
RELAY_SECRET=${RELAY_SECRET}
ROLLOUT_JOB=${JOBTAG}
SPOOL_DIR=${SPOOL_DIR}
EOF
umask 022
log "endpoint -> ${RUN_DIR}/endpoint.env"

# 4. Startup probe: direct SGLang capture fields, then end-to-end through the relay.
sleep 15
uv run --with httpx python src/rollout/probe.py \
  --local "http://127.0.0.1:${PORT}" --api-key-file "${RUN_DIR}/sglang_api_key" \
  --relay "$RELAY_URL" --relay-secret "$RELAY_SECRET" --spool "$FWD_CAPTURE_DIR" --trial "probe-${JOBTAG}" \
  > "${LOG_DIR}/probe.json" 2> "${LOG_DIR}/probe.err" \
  && log "probe ok -> ${LOG_DIR}/probe.json" || log "PROBE FAILED -> ${LOG_DIR}/probe.err"

# 4b. Weight watcher (optional): LATEST_FILE=<trainer run>/LATEST hot-swaps SGLang to each
#     checkpoint the trainer publishes and bumps weight_version (see src/rollout/weight_watcher.py).
start_watcher() {
  LATEST_FILE="$LATEST_FILE" SGLANG_URL="http://127.0.0.1:${PORT}" \
    API_KEY_FILE="${RUN_DIR}/sglang_api_key" ADMIN_KEY_FILE="${RUN_DIR}/sglang_admin_api_key" \
    WEIGHT_VERSION_FILE="${RUN_DIR}/weight_version" \
    uv run --with httpx python src/rollout/weight_watcher.py >> "${LOG_DIR}/weight_watcher.log" 2>&1 &
  WATCH_PID=$!; PIDS+=("$WATCH_PID")
}
WATCH_PID=""
if [ -n "${LATEST_FILE:-}" ]; then start_watcher; log "weight watcher on ${LATEST_FILE}"; fi

# 5. Dispatcher (optional): DISPATCH_TASKS=task1,task2 runs real MLE-Bench trials
#    (stock openhands-sdk on SAIL) against this job's relay; logs in RUN_DIR/dispatch_logs.
if [ -n "${DISPATCH_TASKS:-}" ]; then
  .venv/bin/python src/rollout/dispatch.py --run-dir "$RUN_DIR" --tasks "$DISPATCH_TASKS" \
    --max-concurrent "${DISPATCH_MAX_CONCURRENT:-2}" --max-iterations "${MAX_ITERATIONS:-200}" \
    --rounds "${DISPATCH_ROUNDS:-1}" --condenser-max-events "${CONDENSER_MAX_EVENTS:-120}" --terminal-max-timeout "${TERMINAL_MAX_TIMEOUT:-1800}" \
    > "${LOG_DIR}/dispatch.log" 2>&1 &
  PIDS+=("$!")
  log "dispatcher started for ${DISPATCH_TASKS}"
fi

# 6. Supervise: restart the forwarder if it dies; end the job if SGLang dies.
log "running"
while true; do
  kill -0 "$SGLANG_PID" 2>/dev/null || { log "sglang exited"; tail -30 "${LOG_DIR}/sglang.log"; exit 1; }
  if ! kill -0 "$FWD_PID" 2>/dev/null; then
    log "forwarder exited; restarting"
    RELAY_WS_URL="wss://${RELAY_URL#https://}/bridge" RELAY_SECRET="$RELAY_SECRET" \
      VLLM_LOCAL_URL="http://127.0.0.1:${PORT}" BACKEND_API_KEY="$API_KEY" \
      CAPTURE_DIR="$FWD_CAPTURE_DIR" WEIGHT_VERSION_FILE="${RUN_DIR}/weight_version" MAX_GEN_TOKENS="$MAX_GEN_TOKENS" \
      uv run --with websockets --with httpx python src/relay/node_forwarder.py >> "${LOG_DIR}/forwarder.log" 2>&1 &
    FWD_PID=$!; PIDS+=("$FWD_PID")
  fi
  if [ -n "$WATCH_PID" ] && ! kill -0 "$WATCH_PID" 2>/dev/null; then log "weight watcher exited; restarting"; start_watcher; fi
  sleep 30
done
