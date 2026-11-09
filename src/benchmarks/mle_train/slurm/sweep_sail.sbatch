#!/usr/bin/env bash
# Baseline sweep: N_ATTEMPTS trials of every task with one model, the agent on SAIL and the model on one Babel GPU.
# Resumable: it survives preemption (Slurm requeues it) and plain resubmission; finished trials are kept.
#
#   agent (training's condensing openhands-sdk) in SAIL --https--> Modal relay --ws--> node_forwarder --> vLLM
#
# Trials run from a node-local copy of the tasks (Harbor hashes every task file, far too slow over NFS).
# Resume works because everything the trials' saved config points at stays the same across restarts: the relay
# app name (so its URL) and its secret (SWEEP_DIR/.relay_secret, mode 600). A restart redeploys the same relay,
# terminates SAIL sandboxes the killed attempt left running, then runs `harbor job resume` on the job directory.
# The relay is stopped whenever the job exits (with -y; without it the stop waits for a confirmation).
#
# Submit from the repo root (once; it requeues itself after preemption):
#   mkdir -p logs
#   TASKS=/data/user_data/jeffc2/datasets/mle-train/harbor sbatch --export=ALL src/benchmarks/mle_train/slurm/sweep_sail.sbatch
# If it ever stops for good (time limit, cancel), submit the same command again: it resumes.
# If tasks change between runs (added, removed or rebuilt), delete JOB_DIR/lock.json (Harbor refuses to resume
# when its recorded task checksums differ) and the trial dirs of changed tasks, then resubmit.
#
# Env: TASKS (Harbor tasks dir, required); SWEEP_NAME (default qwen3.5-9b-k4); SWEEP_DIR (default
#      <TASKS>/../sweeps/<SWEEP_NAME>); MODEL (default Qwen/Qwen3.5-9B); N_ATTEMPTS (4); N_CONCURRENT (20);
#      MAX_ITERATIONS (200), CONDENSER_MAX_EVENTS (120), TERMINAL_MAX_TIMEOUT (1800), MAX_GEN_TOKENS (16384):
#      as in scripts/rollout.sh; AGENT (default src.rollout.openhands_agent:CondensingOpenHandsSDK).
#      N_ATTEMPTS / N_CONCURRENT / agent settings are fixed when the sweep starts (Harbor saves them).
# Results: SWEEP_DIR/jobs/<SWEEP_NAME>/<task>__<id>/verifier/{grade.json,final_score.json,validation_scores.jsonl};
#      per-task table: SWEEP_DIR/summary.tsv (src/benchmarks/mle_train/sweep_summary.py), written at the end.
#SBATCH --job-name=mle-sweep
#SBATCH --account=teruko
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --constraint=VRAM_48GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
cd "$REPO_ROOT"
: "${TASKS:?set TASKS to a directory of Harbor tasks}"
SWEEP_NAME="${SWEEP_NAME:-qwen3.5-9b-k4}"
SWEEP_DIR="${SWEEP_DIR:-$(dirname "$TASKS")/sweeps/${SWEEP_NAME}}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
N_ATTEMPTS="${N_ATTEMPTS:-4}"
N_CONCURRENT="${N_CONCURRENT:-20}"
MAX_ITERATIONS="${MAX_ITERATIONS:-200}"
CONDENSER_MAX_EVENTS="${CONDENSER_MAX_EVENTS:-120}"
TERMINAL_MAX_TIMEOUT="${TERMINAL_MAX_TIMEOUT:-1800}"
MAX_GEN_TOKENS="${MAX_GEN_TOKENS:-16384}"
AGENT="${AGENT:-src.rollout.openhands_agent:CondensingOpenHandsSDK}"
JOBTAG="${SLURM_JOB_ID:-local-$$}"
JOB_DIR="${SWEEP_DIR}/jobs/${SWEEP_NAME}"
PORT="${PORT:-$((20000 + $(printf '%s' "$JOBTAG" | cksum | cut -d' ' -f1) % 20000))}"
export VLLM_CUDA_SIF="${VLLM_CUDA_SIF:-/data/user_data/jeffc2/containers/vllm-openai-v0.31.0.sif}"
export CACHE_DIR="${CACHE_DIR:-/data/user_data/jeffc2/.cache}"
export MODAL_IMAGE_BUILDER_VERSION="${MODAL_IMAGE_BUILDER_VERSION:-2025.06}"  # the repo's modal client (1.5.x)
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"  # harbor imports AGENT from the repo
SCRATCH="/scratch/${USER}/mle-sweep-${JOBTAG}"
# Harbor hashes every file of every task (the job lock, and again for each trial). Over NFS that takes hours for
# the image tasks, so the trials run from a node-local copy at a FIXED path: the saved job config points there, and
# a restart on another node recreates it before resuming.
LOCAL_TASKS="/scratch/${USER}/mle-sweep-tasks/${SWEEP_NAME}"
RELAY_APP="mle-sweep-$(printf '%s' "$SWEEP_NAME" | tr -c 'a-z0-9-' '-' | cut -c1-40)"
PY="${REPO_ROOT}/.venv/bin/python"
MODAL="${REPO_ROOT}/.venv/bin/modal"
mkdir -p "$SWEEP_DIR" "$SCRATCH"
log() { echo "[sweep $(date '+%m-%d %H:%M:%S')] $*"; }
log "job=${JOBTAG} restart=${SLURM_RESTART_COUNT:-0} host=$(hostname) sweep=${SWEEP_DIR}"
if [ -f "${SWEEP_DIR}/DONE" ]; then log "sweep already finished (${SWEEP_DIR}/DONE)"; exit 0; fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PIDS=()
cleanup() {
  trap - EXIT TERM INT
  log "shutting down; stopping relay ${RELAY_APP}"
  for p in "${PIDS[@]}"; do pkill -TERM -P "$p" 2>/dev/null || true; kill "$p" 2>/dev/null || true; done
  "$MODAL" app stop -y "$RELAY_APP" >/dev/null 2>&1 || true
  rm -rf "$SCRATCH"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

# 0. Copy the tasks to local disk (8 tasks at a time; rsync only updates a copy already on this node), while vLLM
#    loads. Waited for before the trials start.
mkdir -p "$LOCAL_TASKS"
# Tasks no longer in TASKS (e.g. excluded by an audit) leave the local copy too, so a resume skips them.
for d in "$LOCAL_TASKS"/*/; do [ -d "$d" ] && [ ! -d "${TASKS}/$(basename "$d")" ] && rm -rf "$d"; done
( ls "$TASKS" | xargs -P 8 -I{} rsync -a --delete "${TASKS}/{}/" "${LOCAL_TASKS}/{}/" \
    && echo ok > "${SCRATCH}/copy_done" ) > "${SWEEP_DIR}/copy-${JOBTAG}.log" 2>&1 &
COPY_PID=$!
log "copying tasks to ${LOCAL_TASKS}"

# 1. vLLM on this node's GPU (serve params from configs/serve/<MODEL>.json).
DATA_PARALLEL=1 TENSOR_PARALLEL=1 PORT="$PORT" BASE_DIR="$SCRATCH" MODEL="$MODEL" \
  bash scripts/serve/serve_vllm.sh > "${SWEEP_DIR}/vllm-${JOBTAG}.log" 2>&1 &
PIDS+=("$!")
log "vLLM starting on :${PORT}"
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  kill -0 "${PIDS[0]}" 2>/dev/null || { log "vLLM exited during startup"; tail -30 "${SWEEP_DIR}/vllm-${JOBTAG}.log"; exit 1; }
  sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || { log "vLLM not healthy after 30 min"; exit 1; }
log "vLLM healthy"

# 2. The sweep's relay: same app name and secret on every restart, so the URL the saved trials use stays valid.
if [ ! -s "${SWEEP_DIR}/.relay_secret" ]; then (umask 077; openssl rand -hex 16 > "${SWEEP_DIR}/.relay_secret"); fi
RELAY_SECRET="$(cat "${SWEEP_DIR}/.relay_secret")"
DEPLOY_OUT="$(RELAY_APP_NAME="$RELAY_APP" RELAY_SECRET_NAME="$RELAY_APP" RELAY_SECRET="$RELAY_SECRET" \
  "$MODAL" deploy src/relay/public_api.py 2>&1)" || { echo "$DEPLOY_OUT"; log "relay deploy failed"; exit 1; }
RELAY_URL="$(printf '%s' "$DEPLOY_OUT" | grep -oE 'https://[a-zA-Z0-9._-]+\.modal\.run' | head -1)"
[ -n "$RELAY_URL" ] || { echo "$DEPLOY_OUT"; log "could not parse relay URL"; exit 1; }
if [ -s "${SWEEP_DIR}/relay_url" ] && [ "$(cat "${SWEEP_DIR}/relay_url")" != "$RELAY_URL" ]; then
  log "relay URL changed ($(cat "${SWEEP_DIR}/relay_url") -> ${RELAY_URL}); saved trials can't reach it"; exit 1
fi
echo "$RELAY_URL" > "${SWEEP_DIR}/relay_url"
log "relay ${RELAY_APP}: ${RELAY_URL}"

# 3. Forwarder: bridge to vLLM and cap each call at MAX_GEN_TOKENS (no spool: results come from the verifier).
RELAY_WS_URL="wss://${RELAY_URL#https://}/bridge" RELAY_SECRET="$RELAY_SECRET" VLLM_LOCAL_URL="http://127.0.0.1:${PORT}" \
  MAX_GEN_TOKENS="$MAX_GEN_TOKENS" "$PY" src/relay/node_forwarder.py > "${SWEEP_DIR}/forwarder-${JOBTAG}.log" 2>&1 &
PIDS+=("$!")
for _ in $(seq 1 30); do
  curl -sf -H "Authorization: Bearer ${RELAY_SECRET}" "${RELAY_URL}/v1/models" >/dev/null 2>&1 && break
  sleep 5
done
curl -sf -H "Authorization: Bearer ${RELAY_SECRET}" "${RELAY_URL}/v1/models" >/dev/null \
  || { log "relay -> vLLM check failed"; tail -20 "${SWEEP_DIR}/forwarder-${JOBTAG}.log"; exit 1; }
log "relay -> forwarder -> vLLM ok"

# 4. Run, or resume after a restart. Harbor runs in the background and the script `wait`s for it: bash runs the TERM
#    trap (relay stop) only between commands, and a `wait` is interrupted by the signal where a foreground child is
#    not.
wait "$COPY_PID" || true
[ -f "${SCRATCH}/copy_done" ] || { log "copying tasks failed (see ${SWEEP_DIR}/copy-${JOBTAG}.log)"; exit 1; }
log "tasks copied ($(du -sh "$LOCAL_TASKS" | cut -f1))"
run_harbor() {
  "${REPO_ROOT}/.venv/bin/harbor" "$@" > "${SWEEP_DIR}/harbor-${JOBTAG}.log" 2>&1 &
  PIDS+=("$!")
  wait "$!"
}
if [ -f "${JOB_DIR}/config.json" ]; then
  # Before resuming: (1) Harbor saves the job config with LLM_API_KEY redacted, so a resume would send the mask and
  # every trial would get 401 "bad relay token": write the real relay secret back (the file is then made private).
  # (2) Trials that ended because the job was killed (exit 143, cancelled) or because the relay refused them are not
  # results: delete them so the resume reruns them. (3) Terminate the SAIL sandboxes of unfinished trials.
  RELAY_SECRET="$RELAY_SECRET" "$PY" - "$JOB_DIR" <<'PY'
import json, os, shutil, subprocess, sys
job = sys.argv[1]
cfg_path = os.path.join(job, "config.json")
cfg = json.load(open(cfg_path))
for agent in cfg.get("agents", []):
    if isinstance(agent.get("env"), dict) and "LLM_API_KEY" in agent["env"]:
        agent["env"]["LLM_API_KEY"] = os.environ["RELAY_SECRET"]
tmp = cfg_path + ".tmp"
with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
    json.dump(cfg, f, indent=4)
os.replace(tmp, cfg_path)


def not_a_result(d):
    try:
        e = json.load(open(os.path.join(d, "result.json"))).get("exception_info") or {}
    except Exception:
        return False
    msg = str(e.get("exception_message", ""))
    if e.get("exception_type") in ("CancelledError", "KeyboardInterrupt") or "(exit 143)" in msg:
        return True
    try:
        log = open(os.path.join(d, "agent", "openhands_sdk.txt"), errors="replace").read()[-20000:]
    except Exception:
        log = ""
    return "bad relay token" in log or "LLMAuthenticationError" in log


trials = [os.path.join(job, d) for d in os.listdir(job) if "__" in d and os.path.isdir(os.path.join(job, d))]
unfinished = [d for d in trials if not os.path.exists(os.path.join(d, "result.json"))]
redo = [d for d in trials if os.path.exists(os.path.join(d, "result.json")) and not_a_result(d)]
want = {"harbor-" + os.path.basename(d).replace("__", "-") + "-env" for d in unfinished + redo}
try:
    out = subprocess.run([".venv/bin/sail", "box", "list", "--json"], capture_output=True, text=True).stdout
    for b in json.loads(out).get("data", []):
        if b.get("name") in want and str(b.get("status", "")).lower() not in ("terminated", "terminating"):
            subprocess.run([".venv/bin/sail", "box", "terminate", b["sailbox_id"]], capture_output=True, text=True)
            print(f"terminated leftover sandbox {b['name']}")
except Exception as e:
    print(f"sandbox cleanup skipped: {e}")
for d in redo:
    shutil.rmtree(d)
print(f"resume prep: secret restored; {len(redo)} interrupted/refused trials removed for rerun; "
      f"{len(unfinished)} unfinished")
PY
  log "resuming ${JOB_DIR}"
  run_harbor job resume -p "$JOB_DIR" \
    || { log "harbor resume exited non-zero (see ${SWEEP_DIR}/harbor-${JOBTAG}.log)"; exit 1; }
else
  log "starting: ${N_ATTEMPTS} attempts x $(ls -d "${TASKS}"/*/ | wc -l) tasks, ${N_CONCURRENT} at a time"
  run_harbor run -p "$LOCAL_TASKS" -a "$AGENT" -m "litellm_proxy/${MODEL}" \
    -e sail.harbor:SailboxEnvironment -k "$N_ATTEMPTS" -n "$N_CONCURRENT" \
    --ae "LLM_API_KEY=${RELAY_SECRET}" --ae "LLM_BASE_URL=${RELAY_URL}/sessions/${SWEEP_NAME}/v1" \
    --ae "LLM_MODEL=litellm_proxy/${MODEL}" --ae "MAX_ITERATIONS=${MAX_ITERATIONS}" \
    --ae "CONDENSER_MAX_EVENTS=${CONDENSER_MAX_EVENTS}" --ae "TERMINAL_MAX_TIMEOUT=${TERMINAL_MAX_TIMEOUT}" \
    --job-name "$SWEEP_NAME" -o "${SWEEP_DIR}/jobs" -y --quiet \
    || { log "harbor run exited non-zero (see ${SWEEP_DIR}/harbor-${JOBTAG}.log)"; exit 1; }
fi

# 5. Summary.
"$PY" src/benchmarks/mle_train/sweep_summary.py --job-dir "$JOB_DIR" --tasks "$TASKS" --out "${SWEEP_DIR}/summary.tsv"
touch "${SWEEP_DIR}/DONE"
rm -rf "$LOCAL_TASKS"  # copies on nodes of preempted attempts stay until that node's /scratch is cleaned
log "finished"
