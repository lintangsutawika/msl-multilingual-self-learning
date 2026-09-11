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
the same coding problems across Python, C++, Go, and Java.

The benchmark uses:

* problem statements and canonical Python tests from
  [`newfacade/LeetCodeDataset`](https://huggingface.co/datasets/newfacade/LeetCodeDataset)
* native language-specific function entrypoints from
  [Doocs LeetCode](https://github.com/doocs/leetcode)
* one shared Python judging layer for all four languages
* native execution adapters for Python, C++, Go, and Java
* a common `solution.json` submission format so different agents can be
  evaluated on the same Harbor tasks

The strict multilingual intersection contains 182 problems. Problem 3319 is
currently omitted because tree/object transport is not yet supported, leaving
181 problems × 4 languages = 724 Harbor tasks.

### Generate the Harbor tasks

Generated Harbor tasks are not committed to Git. Regenerate them locally with:

```bash
uv run python -m \
  msl_multilingual_self_learning.benchmark.leetcode.unified_tasks \
  --output benchmarks/leetcode/tasks \
  --skip-unsupported
```

This creates one Harbor task per problem-language pair, for example:

```text
benchmarks/leetcode/tasks/
├── 3243-python/
├── 3243-cpp/
├── 3243-go/
├── 3243-java/
└── ...
```

All agents should run against this same generated task set so they use the same
problem statements, native entrypoints, and verifier.

### Run with SimpleCodeAgent and vLLM

Start an OpenAI-compatible vLLM server separately. For example, if
`Qwen/Qwen3.5-9B` is served at port 8000:

```bash
MODEL=Qwen/Qwen3.5-9B \
MODEL_BASE_URL=http://127.0.0.1:8000/v1 \
scripts/eval/leetcode.sh
```

By default, `scripts/eval/leetcode.sh` uses:

```text
benchmarks/leetcode/tasks
```

and runs `SimpleCodeAgent`.

`SimpleCodeAgent` requests source code from the model and writes a common
submission format:

```json
{
  "language": "cpp",
  "code": "..."
}
```

The shared verifier reads this `solution.json`, executes the submitted source in
the requested native language, and applies the original Python correctness
tests.

Compilation errors, runtime errors, wrong answers, and passing solutions are
recorded as evaluation outcomes rather than repaired automatically.

### Shared benchmark across agents

The LeetCode benchmark is intended to be agent-independent.


## Knobs (env vars)

| Var                     | Default                 | Meaning                                                                   |
| ----------------------- | ----------------------- | ------------------------------------------------------------------------- |
| `MODEL`                 | *(required)*            | litellm model id, e.g. `openai/<served-name>` or `anthropic/claude-...`.  |
| `MODEL_BASE_URL`        | —                       | OpenAI-compatible base URL (vLLM). Omit for a native provider.            |
| `MODEL_API_KEY`         | `dummy`                 | Key for that server/provider.                                             |
| `ENV`                   | `singularity`           | `singularity` (on-node, via harbor-singularity-hpc) | `docker` | `modal`. |
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
