#!/usr/bin/env bash
# Oracle check of built Harbor tasks (no GPU, no LLM): the oracle submits the sample submission and calls validate
# once, then the verifier grades the private split. Every task should show a valid public score and a small reward.
# Works with sbatch or plain bash. Run from the repo root.
#
#   TASKS=~/mle-train/harbor bash src/benchmarks/mle_train/slurm/oracle_check.sbatch
#   sbatch --export=ALL,TASKS=... [cluster flags] src/benchmarks/mle_train/slurm/oracle_check.sbatch
#
# Env: TASKS (required); INCLUDE (a train.<source>.txt list from `pipeline.py export`); LIMIT (max tasks);
#      N (concurrent trials, default 6); HARBOR (default: harbor on PATH);
#      HARBOR_ENV_ARGS (Harbor environment flags; default: Harbor's Docker environment). On clusters without Docker,
#      e.g. CMU Babel with Singularity:
#        HARBOR_ENV_ARGS="-e harbor_singularity_hpc.environment:SingularityWritableEnvironment
#                         --ek singularity_no_mount=home,tmp --environment-build-timeout-multiplier 4"
#SBATCH --job-name=mle-train-oracle
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
: "${TASKS:?set TASKS to a directory of Harbor tasks}"
HARBOR=${HARBOR:-harbor}
JOBS=${JOBS:-$(dirname "$TASKS")/jobs}
TAG=$(basename "${INCLUDE:-$TASKS}" .txt)
JOB=oracle-${TAG//./-}-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}
INC=()
[[ -n "${INCLUDE:-}" ]] && while read -r t; do [[ -n "$t" ]] && INC+=(-i "$t"); done < "$INCLUDE"
export SAIL_API_KEY=${SAIL_API_KEY:-unused}
# shellcheck disable=SC2086  # HARBOR_ENV_ARGS is a list of flags
"$HARBOR" run -p "$TASKS" -a oracle ${HARBOR_ENV_ARGS:-} -n "${N:-6}" "${INC[@]}" ${LIMIT:+-l "$LIMIT"} \
    -o "$JOBS" --job-name "$JOB" -y --quiet
for T in "$JOBS/$JOB"/*/; do
    printf '%-60s %s | %s\n' "$(basename "$T")" "$(grep -h 'score=' "$T"/agent/oracle.txt 2>/dev/null | tail -n 1)" \
        "$(tail -n 1 "$T"/verifier/test-stdout.txt 2>/dev/null)"
done
