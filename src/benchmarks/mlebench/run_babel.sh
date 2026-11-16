#!/usr/bin/env bash
# Run MLE-bench Harbor tasks built with --mount-data on a Babel compute node (Singularity, no Docker/SAIL).
#
#   src/benchmarks/mlebench/run_babel.sh TASK_DIR [TASK_DIR ...]
#
# Babel's singularity.conf mounts host filesystems (hostfs: /home autofs with every user's NFS home, ...) wherever the
# image has the directory, and under --fakeroot the agent reads them as us; so hostfs and bind-paths are turned off.
#
# One `harbor run` per task: its public data (task.toml [metadata] public_data) is bind-mounted read-only at
# /workspace/public; test answers stay in tests/ (uploaded only for the final grade). Images and sandboxes are built
# on node-local scratch (harbor-singularity-hpc bakes the Dockerfile layers once per node).
#
# Env: AGENT (default oracle; e.g. src.rollout.openhands_agent:CondensingOpenHandsSDK), JOBS_DIR (default
#      <task parent>/../jobs), JOB_PREFIX (default mlebench), N_ATTEMPTS (default 1),
#      GRADER_PY (host grader's python; default the eval env's), RESUME=1 (fixed job names; resume an existing job, see below),
#      RESUME_LLM_BASE_URL, RESUME_LLM_API_KEY (on resume: the agent's new LLM_BASE_URL, and its API key, which Harbor
#      saves masked),
#      AGENT_ARGS (extra harbor args, e.g. --ae LLM_BASE_URL=... -m ...), SCRATCH (default /scratch/$USER/hsh).
set -uo pipefail
cd "$(dirname "$0")/../../.."
HARBOR=.venv/bin/harbor
AGENT=${AGENT:-oracle}
SCRATCH=${SCRATCH:-/scratch/$USER/hsh}
mkdir -p "$SCRATCH/sandbox" "$SCRATCH/images" "$SCRATCH/tmp" "$SCRATCH/validate"
# The host grader runs in the eval environment (MLE-bench pinned + pandas<3, as in the image).
GRADER_PY=${GRADER_PY:-src/benchmarks/mlebench/eval/.venv/bin/python}
[ -x "$GRADER_PY" ] || uv sync -q --project src/benchmarks/mlebench/eval
export SINGULARITY_TMPDIR="$SCRATCH/tmp" APPTAINER_TMPDIR="$SCRATCH/tmp"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
rc=0
for TASK in "$@"; do
    TASK=$(realpath "$TASK")
    NAME=$(basename "$TASK")
    PUB=$(.venv/bin/python -c 'import sys,tomllib; print(tomllib.load(open(sys.argv[1],"rb"))["metadata"]["public_data"])' "$TASK/task.toml") \
        || { echo "SKIP $NAME: no metadata.public_data (build with --mount-data)"; rc=1; continue; }
    [ -d "$PUB" ] || { echo "SKIP $NAME: public data $PUB missing"; rc=1; continue; }
    JOBS_DIR=${JOBS_DIR:-$(dirname "$(dirname "$TASK")")/jobs}
    # RESUME=1: a fixed job name, so a rerun (after preemption or resubmission) resumes the same Harbor job.
    if [ "${RESUME:-0}" = 1 ]; then JOB="${JOB_PREFIX:-mlebench}-$NAME"; else JOB="${JOB_PREFIX:-mlebench}-$NAME-$(date +%Y%m%d-%H%M%S)"; fi
    MOUNTS=$(printf '{"type":"bind","source":"%s","target":"/workspace/public:ro"}' "$PUB")
    # Validation split: the answers and the signing key stay on the host; validate_host.py grades `validate` calls
    # there, and only its endpoint (url + token) is mounted into the container.
    VAL=$(.venv/bin/python -c 'import sys,tomllib; print(tomllib.load(open(sys.argv[1],"rb"))["metadata"].get("val_answers",""))' "$TASK/task.toml")
    VPID=
    if [ -n "$VAL" ]; then
        # A fixed path per job (the saved job config mounts it); a new token on every start.
        EP="$JOBS_DIR/.endpoints/$JOB"
        mkdir -p "$EP" && chmod 700 "$JOBS_DIR/.endpoints" "$EP" && rm -f "$EP/endpoint"
        "$GRADER_PY" src/benchmarks/mlebench/validate_host.py --competition "$NAME" --answers "$VAL" \
            --key "$TASK/environment/score_key" --endpoint-dir "$EP" --log "$JOBS_DIR/$JOB.validate.jsonl" \
            > "$EP/server.log" 2>&1 &
        VPID=$!
        for _ in $(seq 120); do [ -f "$EP/endpoint" ] || ! kill -0 $VPID 2>/dev/null && break; sleep 1; done
        [ -f "$EP/endpoint" ] || { echo "SKIP $NAME: host grader failed: $(tail -3 "$EP/server.log")"; kill $VPID 2>/dev/null; rc=1; continue; }
        MOUNTS+=$(printf ',{"type":"bind","source":"%s","target":"/etc/msl-validate:ro"}' "$EP")
    fi
    MOUNTS="[$MOUNTS]"
    echo "=== $(date +%T) $NAME  public=$PUB  agent=$AGENT${VAL:+  host-validate=$EP}"
    if [ "${RESUME:-0}" = 1 ] && [ -f "$JOBS_DIR/$JOB/config.json" ]; then
        # Resume: Harbor keeps every finished trial and reruns the rest (no result.json, or a result whose error is
        # one of the -f types: cut short by preemption, or the agent's download failed before it started). Only what
        # changes between starts is updated in the saved configs (the job's and every trial's): the model server's URL
        # (a new port per Slurm job).
        .venv/bin/python - "$JOBS_DIR/$JOB/config.json" "${RESUME_LLM_BASE_URL:-}" "${RESUME_LLM_API_KEY:-}" <<'PY'
import glob, json, os, sys
path, url, key = sys.argv[1], sys.argv[2], sys.argv[3]


def set_url(agent):
    # Harbor saves the API key masked ("****", "[REDACTED]", not consistently), so saved configs never match as
    # read: write the real (non-secret) key back everywhere, with the new URL.
    env = agent.get("env") if isinstance(agent, dict) else None
    if not isinstance(env, dict):
        return
    if url and "LLM_BASE_URL" in env:
        env["LLM_BASE_URL"] = url
    if key and "LLM_API_KEY" in env:
        env["LLM_API_KEY"] = key


def rewrite(p, edit):
    cfg = json.load(open(p))
    edit(cfg)
    tmp = p + ".tmp"
    with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        json.dump(cfg, f, indent=4)
    os.replace(tmp, p)


rewrite(path, lambda c: [set_url(a) for a in c.get("agents", [])])
# Harbor matches the finished trials' saved configs against the job's: they must carry the same URL too.
job = os.path.dirname(path)
for p in glob.glob(f"{job}/*/config.json"):
    rewrite(p, lambda c: set_url(c.get("agent")))
for p in glob.glob(f"{job}/*/result.json"):
    try:
        rewrite(p, lambda c: set_url((c.get("config") or {}).get("agent")))
    except (OSError, ValueError):
        pass  # unreadable result: Harbor reruns that trial anyway
PY
        # Harbor's lock.json also records every trial's agent settings (old URL, masked key) and refuses a job whose
        # resolved lock differs; it is only that consistency check, so drop it and let Harbor write a fresh one.
        rm -f "$JOBS_DIR/$JOB/lock.json"
        echo "    resuming $JOBS_DIR/$JOB ($(find "$JOBS_DIR/$JOB" -mindepth 2 -maxdepth 2 -name result.json | wc -l) trial results so far)"
        "$HARBOR" job resume -p "$JOBS_DIR/$JOB" -f CancelledError -f NetworkConnectionError || rc=1
        [ -n "$VPID" ] && { kill $VPID 2>/dev/null; wait $VPID 2>/dev/null; }
        continue
    fi
    # shellcheck disable=SC2086  # AGENT_ARGS is a list of flags
    "$HARBOR" run -p "$TASK" -a "$AGENT" -k "${N_ATTEMPTS:-1}" -n "${N_ATTEMPTS:-1}" \
        -e harbor_singularity_hpc.environment:SingularityWritableEnvironment \
        --ek singularity_no_mount=home,tmp,bind-paths,hostfs \
        --ek "singularity_sandbox_dir=$SCRATCH/sandbox" --ek "singularity_image_cache_dir=$SCRATCH/images" \
        --mounts "$MOUNTS" ${AGENT_ARGS:-} \
        -o "$JOBS_DIR" --job-name "$JOB" -y --quiet || rc=1
    [ -n "$VPID" ] && { kill $VPID 2>/dev/null; wait $VPID 2>/dev/null; }
done
exit $rc
