#!/usr/bin/env bash
# Trainer job of the lifelong loop: MILES PPO (actor + critic, 8 GPUs) on episodes the
# rollout job records in the shared spool. Runs no SGLang; reads the spool, writes
# checkpoints (and HF exports) to MSL_STORE/RUN_NAME.
#
#   qsub -N trainer -j oe -o logs/ -P gai51740 -q R9920261300 -v RTYPE=rt_HF,MILES_SIF=<sif> \
#        -l select=1 -l walltime=24:00:00 scripts/trainer.sh
#
# Entry point: src/trainer/run_train.py, bind-mounted over MILES's example run.py; the
# rollout function is src/trainer/spool_rollout.py.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${PBS_O_WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT"
: "${MILES_SIF:?set MILES_SIF}"
# Checkpoints live on group storage (/home has a 2 TiB quota; one actor+critic save is ~260 GB).
MSL_STORE="${MSL_STORE:-/groups/gai51740/aci18914wh/msl/runs}"
RUN_NAME="${RUN_NAME:-trainer-${PBS_JOBID%%.*}}"
export SAVE="${SAVE:-${MSL_STORE}/${RUN_NAME}/checkpoints}"
export MILES_SAVE_HF="${MILES_SAVE_HF:-${MSL_STORE}/${RUN_NAME}/hf/step_{rollout_id}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
mkdir -p "${MSL_STORE}/${RUN_NAME}"
GBS="${GBS:-4}"
SPOOL_DIR="${SPOOL_DIR:-${MSL_STORE}/rollout/spool}"
PROG="${REPO_ROOT}/logs/trainer-${PBS_JOBID%%.*}.progress"
LOCAL="${PBS_LOCALDIR:-$REPO_ROOT/logs/trainer-local}"; mkdir -p "$LOCAL"
NODE_IP="$(hostname -i 2>/dev/null | awk '{print $1}')"; NODE_IP="${NODE_IP:-127.0.0.1}"
export MASTER_ADDR="$NODE_IP" TMPDIR="$LOCAL" RAY_TMPDIR="$LOCAL"
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1 RAY_object_store_memory=5368709120 RAY_raylet_startup_timeout=300
echo "[trainer] start $(date) run=$RUN_NAME save=$SAVE spool=$SPOOL_DIR node_ip=$NODE_IP" > "$PROG"
run_trainer() {
  singularity exec --nv --writable-tmpfs --ipc \
    --bind "${REPO_ROOT}/src:/mnt/msl_src" \
    --bind "${REPO_ROOT}/src/trainer/run_train.py:/root/miles/examples/experimental/harbor/run.py" \
    --bind "${REPO_ROOT}/runs:${REPO_ROOT}/runs" \
    --bind "${MSL_STORE}:${MSL_STORE}" \
    --env "MSL_REPO=${REPO_ROOT}" \
    --env "GLOBAL_BATCH_SIZE=${GBS}" --env "NUM_GPUS=8" --env "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" \
    --env "ONLINE_MODE=1" --env "SPOOL_DIR=${SPOOL_DIR}" \
    --env "MODEL=Qwen/Qwen3.5-9B" \
    --env "MASTER_ADDR=${NODE_IP}" --env "TMPDIR=${LOCAL}" --env "RAY_TMPDIR=${LOCAL}" \
    --env "RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1" --env "RAY_object_store_memory=5368709120" --env "RAY_raylet_startup_timeout=300" \
    "${MILES_SIF}" bash -c "set -euo pipefail
test -d ${SPOOL_DIR} && echo '[trainer] spool visible' >> ${PROG}
cd /root/miles/examples/experimental/harbor
PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/src:/root/miles:/root/Megatron-LM:/root/miles/examples/experimental/harbor:/root/miles/examples/swe-agent-harbor-docker python3 run.py --mode train --save ${SAVE} --save-interval 1 2>&1 | tee -a ${PROG}"
}
ATTEMPTS="${ATTEMPTS:-5}"
for attempt in $(seq 1 "$ATTEMPTS"); do
  echo "[trainer] attempt $attempt $(date)" >> "$PROG"
  if run_trainer; then echo "[trainer] SUCCEEDED on attempt $attempt" >> "$PROG"; exit 0; fi
  echo "[trainer] attempt $attempt failed; backoff 60s" >> "$PROG"
  sleep 60
done
echo "[trainer] all attempts failed" >> "$PROG"; exit 1
