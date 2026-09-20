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

This repository also contains a multilingual LeetCode benchmark for evaluating
the same coding problems across nine programming languages:
* Python
* C++
* Go
* Java
* Rust
* JavaScript
* TypeScript
* PHP
* Ruby

The benchmark uses official LeetCode `codeSnippets` metadata to define the
language-specific function interfaces and a shared Harbor verifier to evaluate
solutions across languages.

Generated Harbor task directories are not committed to Git. They should be
regenerated locally from the committed execution dataset and split manifests.

### Evaluation sets
The benchmark defines three evaluation sets:

* a1: all candidate problems supporting all 9 target languages (3079 problems; full execution dataset not prepared yet)
* a2: all candidate problems supporting Python, C++, Go, and Java (3101 problems; full execution dataset not prepared yet)
* b: a1 intersected with the newfacade/LeetCodeDataset test split (201 problems)
Set B contains 201 selected problem IDs. Problem 3319 is currently excluded from runnable Harbor tasks because tree transport is not yet supported. This leaves 200 runnable problems across 9 languages, for a total of 1800 Harbor tasks.

### Prepare Harbor tasks
Prebuilt language-specific Singularity images can be supplied with:

```bash
export LEETCODE_IMAGE_DIR=/path/to/msl-images
```

The directory should contain:

```text
leetcode-python.sif
leetcode-cpp.sif
leetcode-go.sif
leetcode-java.sif
leetcode-rust.sif
leetcode-javascript.sif
leetcode-typescript.sif
leetcode-php.sif
leetcode-ruby.sif
```

Prepare Set B with:

```bash
uv run python -m src.benchmarks.leetcode \
  --set b \
  --skip-unsupported
```

This writes the generated Harbor tasks to:

```text
benchmarks/leetcode/tasks-b/
```

### SimpleCodeAgent
The benchmark can be evaluated with the repository's `SimpleCodeAgent` through
the standard evaluation script.

For example:

```bash
AGENT="msl_multilingual_self_learning.agents.simple_code_agent:SimpleCodeAgent" \
MODEL="openai/Qwen3.5-9B" \
MODEL_BASE_URL="http://127.0.0.1:8000/v1" \
MODEL_API_KEY=dummy \
TASK_PATH="benchmarks/leetcode/tasks-b" \
scripts/eval/run.sh
```

`MAX_TOKENS` controls the model completion limit:

```bash
MAX_TOKENS=8192
```

and `AGENT_TIMEOUT_MULT` controls Harbor's agent timeout multiplier:

```bash
AGENT_TIMEOUT_MULT=1.0
```

`SimpleCodeAgent` requests source code from the model and writes a common
submission format:

```json
{
  "language": "cpp",
  "code": "..."
}
```

The verifier compiles or executes the submitted source in the requested language and applies the benchmark correctness tests. Compilation errors, runtime errors, wrong answers, and passing solutions are recorded as evaluation outcomes rather than repaired automatically.

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
