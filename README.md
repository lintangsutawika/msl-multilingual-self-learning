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

## Knobs (env vars)

| Var | Default | Meaning |
|---|---|---|
| `MODEL` | *(required)* | litellm model id, e.g. `openai/<served-name>` or `anthropic/claude-...`. |
| `MODEL_BASE_URL` | — | OpenAI-compatible base URL (vLLM). Omit for a native provider. |
| `MODEL_API_KEY` | `dummy` | Key for that server/provider. |
| `ENV` | `singularity` | `singularity` (on-node, via harbor-singularity-hpc) \| `docker` \| `modal`. |
| `N_CONCURRENT` | `4` | Parallel trials. |
| `N_TASKS` | *(all 300)* | Subset size for smoke tests (`-l`). |
| `DATASET` | `swebench_multilingual` | Harbor dataset id (resolves against the default hub registry). |
| `JOBS_DIR` | `jobs` | Output directory. |
| `SINGULARITY_NO_MOUNT` | `home,tmp` | Keep `bind-paths` so the container has DNS (needed for in-container pip). |
| `SINGULARITY_CACHE_DIR` | *(node-local)* | Set a shared-FS path to persist the .sif cache across jobs/resume chunks. |

## Notes

- **Environments.** SWE-bench Multilingual tasks are Dockerfile-defined
  (`FROM swebench/sweb.eval.x86_64.<instance>`) with no `docker_image` in
  `task.toml`. `ENV=singularity` handles that via `harbor-singularity-hpc` (Dockerfile
  `FROM` fallback + writable sandbox for FUSE-restricted nodes); `ENV=docker`/`modal`
  build the Dockerfile natively.
- **Model server.** This scaffold does not serve a model — point `MODEL_BASE_URL` at
  your own endpoint. If you serve a tool-calling model with vLLM, set the matching
  `--tool-call-parser` on the server side (a serve concern, not harbor).
- **Resume.** Re-running the same `JOB_NAME` into the same `JOBS_DIR` resumes; harbor
  stores the environment `import_path` in the job config, so a resumed chunk reloads the
  same Singularity class (the package must be installed on that node).
