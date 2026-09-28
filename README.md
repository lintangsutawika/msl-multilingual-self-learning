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

A benchmark for comparing the same models across 9 languages on the same LeetCode problems: Python, C++, Go, Java, Rust, JavaScript, TypeScript, PHP and Ruby. Each (problem, language) pair is one Harbor task. 
Every language is judged by the same test cases, and the verifier is built so that languages start on equal terms (see [How submissions are judged](#how-submissions-are-judged)).

Generated task directories are not committed; generate them from the committed dataset (`benchmarks/leetcode/data/`) as below. **Task directories embed the verifier, so regenerate them after pulling harness changes.**

### Quick start (Set B)

```bash
uv sync

# 1. Build the nine language images (once, or after a Dockerfile changes).
LEETCODE_IMAGE_DIR=/path/to/images scripts/build/build_leetcode_images.sh

# 2. Generate the Set B tasks; each task.toml points at its language image.
LEETCODE_IMAGE_DIR=/path/to/images \
uv run python -m src.benchmarks.leetcode --set b --skip-unsupported
#    -> benchmarks/leetcode/tasks-b/ (1773 tasks), plus exclusions.json and
#       dropped_tests.json describing what was left out

# 3. Run an agent over them (mini-swe-agent is the default AGENT).
TASK_PATH=benchmarks/leetcode/tasks-b \
MODEL=openai/Qwen/Qwen3.5-9B MODEL_BASE_URL=http://127.0.0.1:8000/v1 \
scripts/eval/run.sh
```

`scripts/eval/run.sbatch` serves the model with vLLM and runs the same eval on one node (`TASKS=benchmarks/leetcode/tasks-b`). Instead of step 1, generation
can build the images itself with `--prebuild-sif`.

### Evaluation sets

* b: problems supporting all 9 languages that intersected with the newfacade/LeetCodeDataset test split (201 problems)

Of Set B's 201 problems, 197 are runnable (1773 tasks): 3319 needs tree transport, which is not supported yet, and 3266, 3387 and 3405 have fewer than 10 valid test cases (see [Invalid test cases](#invalid-test-cases)).

### Images

Each language's image is defined by its task template's Dockerfile, the single source for both `scripts/build/build_leetcode_images.sh` and `--prebuild-sif`:

```text
src/benchmarks/leetcode/task-template-<language>/environment/Dockerfile
```

The build script writes `leetcode-<language>.sif` for all nine languages (or only the languages named as arguments) to `$LEETCODE_IMAGE_DIR` (default `/data/user_data/$USER/msl-images/`), replacing existing images. The JavaScript and TypeScript images include the libraries LeetCode provides to
those languages (`@datastructures-js/priority-queue` v6, `queue`, `deque`
and `lodash`), available to solutions as globals. Every image also contains
what Harbor's Singularity bootstrap needs (its `/opt/harbor-server` venv with
uvicorn/fastapi, `tmux`, `asciinema`); without them each trial downloads these
at start and, under load, exceeds the environment start timeout. Rebuild
images built before this change.

### Running an agent

Tasks work with any Harbor agent. The agent's job is to leave its solution in `/workspace/solution.<ext>` (e.g. `solution.py`, `Solution.java`), which the verifier packages, or to write `/workspace/solution.json` directly:

```json
{"language": "cpp", "code": "..."}
```

The repository's `SimpleCodeAgent` asks the model once for the source file and writes `solution.json`:

```bash
AGENT="msl_multilingual_self_learning.agents.simple_code_agent:SimpleCodeAgent" \
MODEL="openai/Qwen/Qwen3.5-9B" \
MODEL_BASE_URL="http://127.0.0.1:8000/v1" \
MODEL_API_KEY=dummy \
TASK_PATH="benchmarks/leetcode/tasks-b" \
scripts/eval/run.sh
```

`MAX_TOKENS` caps the model's completion and `AGENT_TIMEOUT_MULT` scales Harbor's agent timeout (task timeout 300 s).

### Reading results

Each trial's verdict is in `jobs/<job-name>/<task>__<id>/verifier/`:

| File | Contents |
| --- | --- |
| `reward.txt` | `1` for pass, `0` otherwise |
| `status.txt` | `PASS`, `WRONG_ANSWER`, `COMPILE_ERROR`, `RUNTIME_ERROR` or `TIMEOUT` |
| `normalizations.json` | structural adaptations the verifier applied (see below) |
| `compile.txt`, `worker-stderr.txt`, `test-stdout.txt` | compiler output, the solution's stderr/prints, and the failing assertion |

A trial without `normalizations.json` never reached judging: the agent left no submission (e.g. it timed out), and `test-stdout.txt` reports the missing `solution.json`.

### How submissions are judged

Every language is judged by the same Python `check(candidate)` tests: the verifier (`tests/test.py`, from `src/benchmarks/leetcode/judge_runtime.py`) compiles the submission with a per-language runner and sends each call's arguments to it as a JSON line. Models are judged on their algorithm, not on
guessing the runner's contract, so the verifier accepts any reasonable file structure, with LeetCode's usual implicit imports:

* Go: any package name (rewritten to `main`), a user `main()`, and missing or unused standard imports (fixed from compiler errors)
* Rust: the solution's own `struct Solution`, or a free function instead of an `impl Solution` method; `std::collections::*` is in scope
* C++ (C++20): a free function instead of a `Solution` method; `bits/stdc++.h`
* Java: no imports (`java.util.*`, `java.util.function.*`,
  `java.util.stream.*` and `java.math.*` are added) and any `package` line
* PHP: a missing `<?php` tag; JS: `module.exports` or `let`/`const` entry points; 
* Ruby: a `Solution` class instead of a top-level method
* TypeScript: compiled non-strict, targeting ES2022, with options pinned by the verifier

Anything a solution prints goes to stderr and never affects the verdict.
Builds happen in a fresh `/workspace/.leetcode-build/`, so files the agent leaves in `/workspace` (tsconfig.json, Cargo.toml, package.json) are ignored.

The algorithm itself is never repaired: compile errors in the solution's own code, wrong answers and crashes count as failures.

### Invalid test cases

LeetCodeDataset generated its test inputs and recorded the Python reference solution's output without checking them against the problem's constraints, so some cases break the problem's own contract and have no well-defined answer. Generation removes them for every language:

* inputs that violate LeetCode's Constraints section (e.g. `s.length == 30`
  where the problem requires `s.length == t.length`, or queries `[5, 5]` where it requires `l < r`). `src/benchmarks/leetcode/constraints.py` turns each problem's Constraints into checks and records the violating cases in `benchmarks/leetcode/data/invalid_tests.json`;
* values that do not fit a language's declared LeetCode types (board values of 7e9 cannot even be received in Rust/C++/Java's `int`), and expected outputs of `inf`/`-inf`, which no language can return.

Problems left with fewer than 10 valid test cases are excluded. On Set B, 1857 of 20272 test cases are dropped across 146 problems. To rerun the audit (it fetches each problem's LeetCode page for its exact Constraints):

```bash
uv run python -m src.benchmarks.leetcode.constraints --set b --fetch
```

### Refreshing the dataset

To refresh the official LeetCode metadata cache and rebuild the execution
dataset:

```bash
uv run python -m src.benchmarks.leetcode.fetch_leetcode_cache
uv run python -m src.benchmarks.leetcode.generate_dataset \
  --split test \
  --output benchmarks/leetcode/data/leetcode_multilingual_leetcode.jsonl
```

The crawler also records LeetCode's public examples (from the
`exampleTestcaseList` GraphQL field and the statement's `Output:` blocks)
under `public_test_cases`:

```json
[
  {"input": "[2,7,11,15]\n9", "output": "[0,1]"}
]
```

These are separate from `canonical_tests`, which come from
`newfacade/LeetCodeDataset` and drive the verifier. Rerun the constraint audit after rebuilding the dataset.

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
