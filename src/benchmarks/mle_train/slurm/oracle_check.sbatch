#!/usr/bin/env bash
# Oracle check of built Harbor tasks on SAIL sandboxes (no GPU, no LLM): the oracle submits the sample submission and
# calls validate once, then the verifier grades the private split. Every task should show a signed public score
# (score=<s> sig=<hex>) and a small reward. Works with sbatch or plain bash. Run from the repo root.
# SAIL builds each task's image once (cached per organization); the first run of a task takes longer.
#
#   TASKS=~/mle-train/harbor bash src/benchmarks/mle_train/slurm/oracle_check.sbatch
#   sbatch --export=ALL,TASKS=... [cluster flags] src/benchmarks/mle_train/slurm/oracle_check.sbatch
#
# Env: TASKS (required); INCLUDE (a train.<source>.txt list from `pipeline.py export`); LIMIT (max tasks);
#      N (concurrent trials, default 6);
#      HARBOR (default: the repo .venv's harbor, else harbor on PATH);
#      HARBOR_ENV_ARGS (Harbor environment flags; default: SAIL, -e sail.harbor:SailboxEnvironment).
#      SAIL credentials: SAIL_API_KEY in the environment, SAIL_KEY in the repo .env, or a `sail auth login`.
#SBATCH --job-name=mle-train-oracle
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
: "${TASKS:?set TASKS to a directory of Harbor tasks}"
HARBOR=${HARBOR:-$([[ -x .venv/bin/harbor ]] && echo .venv/bin/harbor || echo harbor)}
HARBOR_ENV_ARGS=${HARBOR_ENV_ARGS:--e sail.harbor:SailboxEnvironment}
JOBS=${JOBS:-$(dirname "$TASKS")/jobs}
TAG=$(basename "${INCLUDE:-$TASKS}" .txt)
JOB=oracle-${TAG//./-}-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}
INC=()
[[ -n "${INCLUDE:-}" ]] && while read -r t; do [[ -n "$t" ]] && INC+=(-i "$t"); done < "$INCLUDE"
# SAIL key: environment first, then SAIL_KEY in the repo .env (never printed); else the `sail auth login` config.
if [[ -z "${SAIL_API_KEY:-}" && -f .env ]]; then
    SAIL_API_KEY=$(sed -n 's/^\(export \)\{0,1\}SAIL_\(API_\)\{0,1\}KEY=//p' .env | head -n 1 | tr -d "\"'")
fi
[[ -n "${SAIL_API_KEY:-}" ]] && export SAIL_API_KEY
# shellcheck disable=SC2086  # HARBOR_ENV_ARGS is a list of flags
"$HARBOR" run -p "$TASKS" -a oracle $HARBOR_ENV_ARGS -n "${N:-6}" "${INC[@]}" ${LIMIT:+-l "$LIMIT"} \
    -o "$JOBS" --job-name "$JOB" -y --quiet
for T in "$JOBS/$JOB"/*/; do
    printf '%-60s %s | %s\n' "$(basename "$T")" "$(grep -h 'score=' "$T"/agent/oracle.txt 2>/dev/null | tail -n 1)" \
        "$(tail -n 1 "$T"/verifier/test-stdout.txt 2>/dev/null)"
done
