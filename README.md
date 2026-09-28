# msl-multilingual-self-learning

Multilingual self-learning experiments. This scaffold evaluates
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) on **SWE-bench
Multilingual** (300 tasks across many languages) using
[harbor](https://github.com/harbor-framework/harbor).

## Setup

```bash
uv sync
```

Installs harbor (pinned), mini-swe-agent, and `harbor-singularity-hpc` (the
writable-rootfs Singularity environment for running task containers on a PBS/SLURM
node). See `pyproject.toml` `[tool.uv.sources]` for the git pins.

## Run

Point the agent at any model endpoint and launch:

```bash
# Against an OpenAI-compatible server (e.g. a local vLLM):
MODEL=openai/Qwen/Qwen3-Coder-30B-A3B-Instruct \
MODEL_BASE_URL=http://localhost:8000/v1 MODEL_API_KEY=dummy \
scripts/eval/run.sh

# Or a native provider (set that provider's key in your env):
MODEL=anthropic/claude-sonnet-4-5 ANTHROPIC_API_KEY=... scripts/eval/run.sh
```

Smoke-test on a handful of tasks first:

```bash
N_TASKS=5 N_CONCURRENT=2 MODEL=... MODEL_BASE_URL=... scripts/eval/run.sh
```

Results land under `jobs/<job-name>/`.

## Multilingual LeetCode

A benchmark for comparing the same models across nine languages on the same
LeetCode problems: Python, C++, Go, Java, Rust, JavaScript, TypeScript, PHP
and Ruby. Each (problem, language) pair is one Harbor task. Every language is
judged by the same test cases, and the verifier is built so that languages
start on equal terms (see [How submissions are judged](#how-submissions-are-judged)).

Problems come from the hosted [`neulab/leetcode`](https://huggingface.co/datasets/neulab/leetcode)
dataset (built by `src/benchmarks/leetcode/hf_dataset/build_dataset.py`): one
row per (problem, language) with the official LeetCode interface and the
LeetCodeDataset `check(candidate)` tests. Generated task directories are not
committed. **Task directories embed the verifier, so regenerate them after
pulling harness changes.**

### Quick start (test set)

```bash
uv sync

# 1. Build the nine language images (once, or after a Dockerfile changes).
LEETCODE_IMAGE_DIR=/path/to/images scripts/build/build_leetcode_images.sh

# 2. Generate the test-set tasks; each task.toml points at its language image.
LEETCODE_IMAGE_DIR=/path/to/images \
uv run python -m src.benchmarks.leetcode --set test --skip-unsupported
#    -> benchmarks/leetcode/tasks-test/, plus exclusions.json and
#       dropped_tests.json describing what was left out

# 3. Run an agent over them (mini-swe-agent is the default AGENT).
TASK_PATH=benchmarks/leetcode/tasks-test \
MODEL=openai/Qwen/Qwen3.5-9B MODEL_BASE_URL=http://127.0.0.1:8000/v1 \
scripts/eval/run.sh
```

`scripts/eval/run.sbatch` serves the model with vLLM and runs the same eval on
one node. Instead of step 1, generation can build the images itself with
`--prebuild-sif`.

### Sets

* `--set test`: the 201 problems of the LeetCodeDataset test split that
  support all nine languages. 197 are runnable (1773 tasks): 3319 needs tree
  transport, which is not supported yet, and 3266, 3387 and 3405 have fewer
  than 10 valid test cases (see [Invalid test cases](#invalid-test-cases)).
* `--set train`: the LeetCodeDataset train split problems supporting all nine
  languages (for training; disjoint from the test set).

`--lang` limits which languages become tasks; `--question-id` and `--limit`
select problems.

### Images

Each language's image is defined by its task template's Dockerfile, the
single source for both `scripts/build/build_leetcode_images.sh` and
`--prebuild-sif`:

```text
src/benchmarks/leetcode/task-template-<language>/environment/Dockerfile
```

The build script writes `leetcode-<language>.sif` for all nine languages (or
only the languages named as arguments) to `$LEETCODE_IMAGE_DIR` (default
`/data/user_data/$USER/msl-images/`), replacing existing images.

* The JavaScript and TypeScript images include the libraries LeetCode provides
  to those languages (`@datastructures-js/priority-queue` v6, `queue`, `deque`
  and `lodash`), available to solutions as globals; TypeScript is pinned to 5.x.
* Every image contains what Harbor's Singularity bootstrap needs (its
  `/opt/harbor-server` venv with uvicorn/fastapi, `tmux`, `asciinema`).
  Without them each trial downloads these at start and, under load, exceeds
  the environment start timeout (`EnvironmentStartTimeoutError`). Rebuild
  images built before this change.

### Running an agent

Tasks work with any Harbor agent. The agent's job is to leave its solution in
`/workspace/solution.<ext>` (e.g. `solution.py`, `Solution.java`), or to write
`/workspace/solution.json` directly:

```json
{"language": "cpp", "code": "..."}
```

The repository's `SimpleCodeAgent` asks the model once for the source file
and writes `solution.json`:

```bash
AGENT="msl_multilingual_self_learning.agents.simple_code_agent:SimpleCodeAgent" \
MODEL="openai/Qwen/Qwen3.5-9B" \
MODEL_BASE_URL="http://127.0.0.1:8000/v1" \
MODEL_API_KEY=dummy \
TASK_PATH="benchmarks/leetcode/tasks-test" \
scripts/eval/run.sh
```

`MAX_TOKENS` caps the model's completion and `AGENT_TIMEOUT_MULT` scales
Harbor's agent timeout (task timeout 300 s).

### Reading results

Each trial's verdict is in `jobs/<job-name>/<task>__<id>/verifier/`:

| File | Contents |
| --- | --- |
| `reward.txt` | `1` for pass, `0` otherwise |
| `status.txt` | `PASS`, `WRONG_ANSWER`, `COMPILE_ERROR`, `RUNTIME_ERROR` or `TIMEOUT` |
| `normalizations.json` | structural adaptations the verifier applied (see below) |
| `compile.txt`, `worker-stderr.txt`, `test-stdout.txt` | compiler output, the solution's stderr/prints, and the failing assertion |

A trial without `normalizations.json` never reached judging: the agent left
no submission (e.g. it timed out) or the environment never started.

### How submissions are judged

Every language is judged by the same Python `check(candidate)` tests: the
verifier (each template's `tests/test.py`, identical across languages)
compiles the submission with a per-language runner (`runners.py`) and sends
each call's arguments to it as a JSON line. Models are judged on their
algorithm, not on guessing the runner's contract, so the verifier accepts any
reasonable file structure, with LeetCode's usual implicit imports:

* Go: any package name (rewritten to `main`), a user `main()`, and missing or
  unused standard imports (fixed from compiler errors)
* Rust: the solution's own `struct Solution`, or a free function instead of an
  `impl Solution` method; `std::collections::*` is in scope
* C++ (C++20): a free function instead of a `Solution` method; `bits/stdc++.h`
* Java: no imports (`java.util.*`, `java.util.function.*`,
  `java.util.stream.*` and `java.math.*` are added) and any `package` line
* PHP: a missing `<?php` tag; JS: `module.exports` or `let`/`const` entry
  points; Ruby: a `Solution` class instead of a top-level method
* TypeScript: compiled non-strict, targeting ES2022, with options pinned by the
  verifier; Python: the builtin `pow` is not shadowed by `math.pow`

Anything a solution prints goes to stderr and never affects the verdict.
Builds happen in a fresh `/workspace/.leetcode-build/`, so files the agent
leaves in `/workspace` (tsconfig.json, Cargo.toml, package.json) are ignored.
The algorithm itself is never repaired: compile errors in the solution's own
code, wrong answers and crashes count as failures.

### Invalid test cases

LeetCodeDataset generated its test inputs and recorded the Python reference
solution's output without checking them against the problem's constraints,
so some cases break the problem's own contract and have no well-defined
answer. Generation removes them for every language:

* inputs that violate LeetCode's Constraints section (e.g. `s.length == 30`
  where the problem requires `s.length == t.length`, or queries `[5, 5]` where
  it requires `l < r`). `src/benchmarks/leetcode/constraints.py` turns each
  problem's Constraints into checks and records the violating cases in
  `benchmarks/leetcode/data/invalid_tests_<split>.json`;
* values that do not fit a language's declared LeetCode types (board values
  of 7e9 cannot even be received in Rust/C++/Java's `int`), and expected
  outputs of `inf`/`-inf`, which no language can return.

Which tests are dropped is decided per problem from all nine languages'
interfaces, so it does not depend on `--lang`. Problems left with fewer than
10 valid test cases are excluded. On the test set, 1857 of 20272 test cases
are dropped across 146 problems. To rerun an audit (it fetches each problem's
LeetCode page for its exact Constraints):

```bash
uv run python -m src.benchmarks.leetcode.constraints --split test --fetch
```

## Knobs (env vars)

| Var                     | Default                 | Meaning                                                                   |
| ----------------------- | ----------------------- | ------------------------------------------------------------------------- |
| `MODEL`                 | *(required)*            | litellm model id, e.g. `openai/<served-name>` or `anthropic/claude-...`.  |
| `MODEL_BASE_URL`        | —                       | OpenAI-compatible base URL (vLLM). Omit for a native provider.            |
| `MODEL_API_KEY`         | `dummy`                 | Key for that server/provider.                                             |
| `ENV`                   | `singularity` (on-node, via harbor-singularity-hpc), `docker`, or `modal`. |
| `N_CONCURRENT`          | `4`                     | Parallel trials.                                                          |
| `N_TASKS`               | *(all 300)*             | Subset size for smoke tests (`-l`).                                       |
| `DATASET`               | `swebench_multilingual` | Harbor dataset id (resolves against the default hub registry).            |
| `JOBS_DIR`              | `jobs`                  | Output directory.                                                         |
| `SINGULARITY_NO_MOUNT`  | `home,tmp`              | Keep `bind-paths` so the container has DNS (needed for in-container pip). |
| `SINGULARITY_CACHE_DIR` | *(node-local)*          | Set a shared-FS path to persist the .sif cache across jobs/resume chunks. |

## Notes

* **Environments.** SWE-bench Multilingual tasks are Dockerfile-defined
  (`FROM swebench/sweb.eval.x86_64.<instance>`) with no `docker_image` in
  `task.toml`. `ENV=singularity` handles that via `harbor-singularity-hpc`
  (Dockerfile `FROM` fallback + writable sandbox for FUSE-restricted nodes);
  `ENV=docker`/`modal` build the Dockerfile natively.
* **Model server.** This scaffold does not serve a model — point
  `MODEL_BASE_URL` at your own endpoint. If you serve a tool-calling model with
  vLLM, set the matching `--tool-call-parser` on the server side (a serve
  concern, not harbor).
* **Resume.** Re-running the same `JOB_NAME` into the same `JOBS_DIR` resumes;
  harbor stores the environment `import_path` in the job config, so a resumed
  chunk reloads the same Singularity class (the package must be installed on
  that node).
