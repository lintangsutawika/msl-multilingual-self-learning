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
#      GRADER_PY (host grader's python),
#      AGENT_ARGS (extra harbor args, e.g. --ae LLM_BASE_URL=... -m ...), SCRATCH (default /scratch/$USER/hsh).
set -uo pipefail
cd "$(dirname "$0")/../../.."
HARBOR=.venv/bin/harbor
AGENT=${AGENT:-oracle}
SCRATCH=${SCRATCH:-/scratch/$USER/hsh}
mkdir -p "$SCRATCH/sandbox" "$SCRATCH/images" "$SCRATCH/tmp" "$SCRATCH/validate"
GRADER_PY=${GRADER_PY:-/data/user_data/jeffc2/mlebench-grader-venv/bin/python}  # mlebench rev + pandas<3, as in the image
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
    JOB="${JOB_PREFIX:-mlebench}-$NAME-$(date +%Y%m%d-%H%M%S)"
    MOUNTS=$(printf '{"type":"bind","source":"%s","target":"/workspace/public:ro"}' "$PUB")
    # Validation split: the answers and the signing key stay on the host; validate_host.py grades `validate` calls
    # there, and only its endpoint (url + token) is mounted into the container.
    VAL=$(.venv/bin/python -c 'import sys,tomllib; print(tomllib.load(open(sys.argv[1],"rb"))["metadata"].get("val_answers",""))' "$TASK/task.toml")
    VPID=
    if [ -n "$VAL" ]; then
        EP=$(mktemp -d "$SCRATCH/validate/$NAME.XXXX")
        mkdir -p "$JOBS_DIR"
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
