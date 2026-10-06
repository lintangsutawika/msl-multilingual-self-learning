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

### How the dataset is built and cleaned

`neulab/leetcode` is built once by `src/benchmarks/leetcode/hf_dataset/` and
pushed to Hugging Face; task generation uses its rows as they are.

```bash
git clone https://huggingface.co/datasets/neulab/leetcode ../neulab-leetcode
# Build into the clone. The first run fetches each problem's LeetCode page (one
# request every --rate-delay seconds, ~2 hours; cached in ~/.cache/msl-leetcode/raw,
# so an interrupted run resumes); later runs take ~2 min. It replaces data/,
# reports/ and README.md; --dry-run writes only reports/ (summary.txt, dropped.md).
uv run python -m src.benchmarks.leetcode.hf_dataset.build_dataset --out ../neulab-leetcode --rate-delay 2
# Publish: `add -A` also records the deletion of data files a build no longer writes.
cd ../neulab-leetcode && git add -A && git commit -m "Rebuild" && git push
```

* **Description**: taken from the LeetCode page with exponents (`10^9`, not
  `109`), subscripts (`l_i`), numbered lists and tables kept; images are dropped.
* **Interfaces**: each language's signature from LeetCode's code snippets.
* **Tests**: LeetCodeDataset's `check(candidate)` asserts (inputs plus the
  Python reference's output), minus the ones that are unfair for some language:
  inputs that break the problem's Constraints (`hf_dataset/constraints.py`
  turns each rule into a check; `hf_dataset/clean.py` holds the drop rules), values that do not fit a declared type
  (7e9 in Rust/C++/Java's `int`), and expected `inf`/`nan`.
* **Comparison**: where a list answer may come in any order, or the answer is
  a decimal, the asserts call `answers_match` (defined at the top of the test
  source: top-level order ignored, or 1e-5 tolerance), and
  `canonical_tests.comparison` says which.
* **Extra columns**: `public_tests` (the inputs of LeetCode's Testcase panel,
  one string per case, one JSON value per line in parameter order; no outputs)
  and `hints` (the page's hints as plain text); not shown to the model unless a
  prompt uses them.
* **Dropped problems**: premium, tree/linked-list inputs, in-place answers
  (LeetCodeDataset only checks `== None`), fewer than 10 valid tests, and in
  the test split also several valid answers or text that refers to a figure.

`reports/dropped.md` in the dataset lists every dropped problem and test.
Test split: 191 of 228 problems kept, with 18,070 tests (1,846 dropped).
Train: 2,061 of 2,641 problems kept, with 189,802 tests.
