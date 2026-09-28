# msl-multilingual-self-learning

LeetCode benchmark with 9 languages (python, cpp, go, java, rust, javascript,
typescript, php, ruby), flattening one task per (problem, language). Prep builds a
hosted HF dataset (`neulab/leetcode`); generation renders Harbor tasks from a
per-language template.

## Setup

### Serve a model (vLLM) for evaluation

Build/point a CUDA vLLM Singularity image, then serve the deliberator model:

```bash
# SIF already exists (e.g. north-vllm-cuda-fixed.sif). Serve a model to evaluate.
VLLM_CUDA_SIF=/path/to/vllm-cuda.sif MODEL=Qwen/Qwen3.6-35B-A3B \
TASKS=benchmarks/leetcode/tasks-test RUN=0 \
  sbatch run.sbatch
```

### Prebuild per-language SIFs for HPCs (Modal parity)

Each language template carries a proper `environment/Dockerfile`. Two ways to serve
tasks from a prebuilt image instead of building at runtime:

```bash
# Option A: build SIFs from the templates during generation (per-language).
uv run python -m src.benchmarks.leetcode --set test \
    --prebuild-sif \
    --output-dir benchmarks/leetcode/tasks-test

# Option B: point at already-built SIFs in a directory (named leetcode-<lang>.sif).
uv run python -m src.benchmarks.leetcode --set test \
    --image-dir /path/to/sifs \
    --output-dir benchmarks/leetcode/tasks-test
```

`--prebuild-sif` builds each language SIF from its template Dockerfile and saves it
**in the task output dir** (same place as the tasks; `--prebuild-dir` overrides).
`task.toml [environment].docker_image` is set to the SIF's absolute path.

## Run

```bash
VLLM_CUDA_SIF=/path/to/image.sif \
N_CONCURRENT=16 \
MAX_MODEL_LEN=262144 \
TASKS=benchmarks/leetcode/tasks-test \
MODEL=Qwen/Qwen3.6-35B-A3B \
DATA_PARALLEL=4 TENSOR_PARALLEL=2 \
RUN=0 \
  sbatch run.sbatch
```
