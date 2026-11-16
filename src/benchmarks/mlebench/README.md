# MLE-bench as Harbor tasks

This directory turns [MLE-bench](https://github.com/openai/mle-bench) competitions into Harbor tasks, optionally with a
validation split, and runs them on Babel (Singularity) or SAIL (Docker).

Each task gives the agent `/workspace/public/` (the competition's data) and expects `/workspace/predictions.csv`. Two
grades are recorded:

- **Final grade.** MLE-bench's own grader on MLE-bench's test set, unchanged:
  - `verifier/grade.json`: score, medal and `valid_submission`;
  - `verifier/reward.txt`: the medal tier (3 gold, 2 silver, 1 bronze, 0 none).
- **Validation score (with `--val-split`).** The agent can run `validate predictions.csv` any number of times. It
  scores only the validation rows and never the test rows.

## Validation split (`--val-split`)

MLE-bench made each competition's test set by running the competition's `prepare.py` on Kaggle's train data.
`valsplit.py` runs the same `prepare.py` a second time, on MLE-bench's prepared train, so the validation set is drawn
the same way the test set was:

- MLE-bench's train becomes **train′**;
- the second split's "test" becomes the **validation set**;
- MLE-bench's test set and answers are not touched.

The validation inputs are shuffled into the test inputs, so the agent sees one test set and writes one
`predictions.csv`. `validate` grades the validation rows; the final grade uses only MLE-bench's test rows.

Most competitions use exactly MLE-bench's split. A few need their own handling: renumbered ids, leaks MLE-bench left in
its test files, or no split to repeat. Each such case is documented on its entry in `SPECS` in `valsplit.py`.
`audit_leaks.py` then checks that no test or validation answer can be found in what the agent sees.

## One-time setup

1. **Repo environment.** `uv sync --extra prep` (installs Harbor, harbor-singularity-hpc and MLE-bench in `.venv`).
2. **Grader environment.** `src/benchmarks/mlebench/setup_grader_venv.sh /path/to/mlebench-grader-venv`.
   - It installs MLE-bench at the same revision plus pandas<3.
   - Preparing data, the validation split, the leak audit and the host-side `validate` grader run with it, because
     several of MLE-bench's `prepare.py` scripts break under pandas 3 (the repo's `.venv`). These scripts stop with
     a clear error if run with pandas 3.
3. **Site paths.** `cp src/benchmarks/mlebench/site.env.example src/benchmarks/mlebench/site.env`, then set:
   - `GRADER_PY`: the grader environment's python;
   - for agent sweeps on Babel, `VLLM_CUDA_SIF` and `CACHE_DIR`.

   `site.env` is git-ignored.
4. **Kaggle.**
   - Put `kaggle.json` in `~/.kaggle/` (or set `KAGGLE_CONFIG_DIR`).
   - Accept each competition's rules on kaggle.com. Until you do, its download returns 403.
5. **Data directory.** MLE-bench keeps data in `~/.cache/mle-bench/data`. It runs to hundreds of GB, so make that a
   symlink to a large disk.

Run every command below from the repo root. The Slurm scripts default to `--account=teruko` and the `preempt`
partition; override them on the `sbatch` command line.

## Steps for one competition

Example: `C=spooky-author-identification`; `T=/path/to/tasks` is where the built tasks go.

**1. Download and prepare** with MLE-bench's own scripts. Big competitions need a few times their download size in
free disk while preparing.

```bash
sbatch --export=ALL,EXTRA="--competition $C --delete-zips" src/benchmarks/mlebench/slurm/prepare_queue.sbatch
```

- Without `--competition`, it prepares every competition of `SPLITS` (default `low,medium,high`), smallest first,
  until the disk is full.
- `--delete-zips` removes Kaggle's zip and `raw/` afterwards. They are only needed to prepare the competition again.

**2. Validation split and leak audit** (skip this step if you don't want a validation split):

```bash
$GRADER_PY -m src.benchmarks.mlebench.valsplit --competition $C
$GRADER_PY -m src.benchmarks.mlebench.audit_leaks --competition $C      # must print "1/1 clean"
```

- These read and write the data directory (`<data>/$C/valsplit/`).
- For big competitions, run them in a Slurm job with 4 CPUs and 32–64 GB.

**3. Build the task.** Choose the build for where the task will run.

```bash
# Babel (Singularity): data stays on disk, mounted read-only at run time
uv run python -m src.benchmarks.mlebench --competition $C --val-split --mount-data --output $T
# SAIL (Docker): the prepared data is copied into the task's image
uv run python -m src.benchmarks.mlebench --competition $C --val-split --output $T
```

- The task lands in `$T/mlebench-{low,medium,high}/$C`.
- `--competition` takes a comma-separated list; `--splits low,medium,high` or `--all` select competitions by level.
- Leave out `--val-split` for MLE-bench's plain setup (no `validate`).
- **What `--mount-data` changes:**
  - The data path goes into `task.toml` (`[metadata] public_data`).
  - The test answers go in `tests/`, which Harbor uploads only for the final grade.
  - The validation answers and the score-signing key stay on the host: Babel's rootless Singularity has no real user
    separation, so anything inside the container is readable by the agent.
- **What the SAIL build does instead:** the answers go into the image, readable only by root, and the agent runs as
  an unprivileged user.

**4. Oracle check (Babel).** The oracle submits the sample submission. It should score near chance and must be
accepted by both graders.

```bash
sbatch --cpus-per-task=4 --mem=16G --time=3:00:00 --output=logs/%x-%j.out \
    --wrap "src/benchmarks/mlebench/oracle_check.sh $T/mlebench-low/$C"
```

**5. Run an agent.**

- **Babel, the setup our sweeps use.** One GPU node per task: vLLM serves the model, and N attempts run in
  Singularity. Preempted jobs resume, keeping finished attempts.

  ```bash
  sbatch --export=ALL,TASKS=$T/mlebench-low/$C,SWEEP_NAME=my-sweep,MODEL=Qwen/Qwen3.5-9B,N_ATTEMPTS=4 \
      src/benchmarks/mlebench/slurm/agent_babel.sbatch
  ```

  - The settings are documented in the script's header, including `AGENT` and `HARBOR_EXTRA_ARGS` (for example
    `--agent-timeout-multiplier 2` for 8 h attempts).
  - To run any other Harbor agent on Babel, call `run_babel.sh` directly. It mounts the data, starts the host-side
    `validate` grader and runs `harbor run`:

    ```bash
    AGENT=<module:Class or harbor agent name> AGENT_ARGS="-m <model> --ae KEY=VALUE ..." N_ATTEMPTS=4 \
        src/benchmarks/mlebench/run_babel.sh $T/mlebench-low/$C
    ```

    Tasks built with `--mount-data` need `run_babel.sh` (or the same mounts); a plain `harbor run` can't see their
    data.
- **SAIL.** Tasks built without `--mount-data` are ordinary Harbor tasks, so any Harbor runner works. For example,
  `src/benchmarks/mle_train/slurm/agent_eval_sail.sbatch` runs an agent on SAIL with the model served from Babel.

**Results.**

- Per attempt: `<jobs dir>/<job>/<trial>/verifier/grade.json`, and `predictions.csv.gz` (the agent's final file, kept
  so it can be regraded).
- Every `validate` call is logged in `<jobs dir>/<job>.validate.jsonl` (Babel) or in the trial's
  `verifier/validation_scores.jsonl`.

## Adding a competition's validation split

`valsplit.py` has a `Spec` for 63 of MLE-bench's 75 competitions. Running it on any other competition stops with "no
validation-split spec yet". To add one:

1. Read the competition's `prepare.py` in MLE-bench.
2. List what `prepare.py` reads from `raw/` (`raw=`) and which public entries hold test inputs (`test_side=`).
3. Copy the `Spec` of a competition shaped the same way. The fields are documented on the `Spec` class.
4. Run the split, the audit and the oracle check (steps 2 and 4).

## Files

| File | What it does |
|---|---|
| `main.py`, `adapter.py` | Build tasks from `task-template/` |
| `prepare_queue.py`, `slurm/prepare_queue.sbatch` | Download and prepare competitions within a disk budget |
| `valsplit.py` | Make the validation split (`SPECS`: one entry per competition) |
| `audit_leaks.py` | Check that no answer leaks into what the agent sees |
| `run_babel.sh` | Run tasks on Babel under Singularity (mounts, host grader, resume) |
| `validate_host.py` | The host-side `validate` grader (Babel) |
| `oracle_check.sh` | Run the oracle and show both graders' verdicts |
| `slurm/agent_babel.sbatch` | Agent sweep on Babel: vLLM plus N attempts per task |
| `setup_grader_venv.sh`, `site.env.example` | One-time setup |
| `task-template/` | The Harbor task: Dockerfile, grader (`tests/test.sh`), `validate`, oracle (`solution/solve.sh`) |
