# Multilingual LeetCode evaluation

Evaluate `Qwen/Qwen3.5-9B` through an OpenAI-compatible vLLM endpoint in Harbor,
using native Python, C++, Go, and Java entrypoints and the same Python tests.
Problem 3319 is omitted by default, leaving 181 active problems (724 tasks).

The dataset in `benchmarks/leetcode/data/leetcode_multilingual.jsonl` contains
182 problems from the **test split** of
[newfacade/LeetCodeDataset](https://huggingface.co/datasets/newfacade/LeetCodeDataset).
Each record retains four [Doocs](https://github.com/doocs/leetcode) interfaces,
the original Python `check(candidate)` source, and the original HF parameter names.
Native signatures are not translated. Two Python interfaces supplement missing
Doocs type annotations from HF starter code.

## Setup and oracle validation

```bash
uv sync

# Uses the existing local external/doocs-leetcode checkout for reference code.
.venv/bin/python -m msl_multilingual_self_learning.benchmark.leetcode.unified_tasks \
  --question-id 3243 --question-id 3304 --question-id 3366 \
  --with-oracle --output benchmarks/leetcode/my-validation

TASKS=benchmarks/leetcode/my-validation AGENT=oracle scripts/eval/leetcode.sh
```

Use a fresh output directory when regenerating tasks. Generation refuses to
replace existing tasks, so saved Harbor runs retain their original inputs.
The legacy `generate_harbor_task` and `generate_harbor_smoke` CLI entrypoints now
route to this generator. Old `datasets/`, `harbor_smoke/`, and `generated_tasks/`
artifacts are historical and are not inputs to the new launcher.

The default backend is `harbor-singularity-hpc` on this HPC host. Docker is
available through `ENV=docker`. Every task provisions Python 3, g++, Go, Java,
nlohmann/json, and Gson. Docker uses its Dockerfile; Singularity uses
`environment/files/setup.sh`, because this backend only reads Dockerfile `FROM`
and does not execute Dockerfile build steps. Initial provisioning requires
network access and writable fakeroot support. Runtime versions are printed in
the Harbor trial log. Ubuntu packages currently provide Go 1.22 and Java 21.
For repeated large runs, use a prebuilt `.sif` containing the same dependencies
via the generator's `--image /absolute/path/runtime.sif` option.

## Shared contract

The agent and oracle both submit `/workspace/solution.json`:

```json
{"language": "python", "code": "class Solution:\n    ..."}
```

Every task has the same Python-only test files. Native workers live under
`environment/files/adapters/` and are installed at `/opt/leetcode`; they contain
no assertions or expected answers. The shared Python verifier validates the
envelope, writes the native source,
and compiles the C++/Go/Java adapter once. All four adapters expose a persistent
JSON-lines worker: each line contains an ordered argument array; each response
contains the return value. The Python judge executes the original
`check(candidate)` unchanged. Each candidate call binds the original HF
parameter names and sends the actual arguments to the native function, so loops
and dynamically computed test arguments work. A solution instance persists
between calls, as in `Solution().method`.

C++ `char` and Go `byte` returns become strings. Go nil slices become empty
arrays. Native parameter names and types remain unchanged. Doocs oracle snippets
receive only language boilerplate (Go package/imports and Java standard imports);
solution logic is not translated. Generated model code must provide its imports
and, for Go, `package main`.

The verifier writes `reward.txt`, `status.txt`, `compile.txt` when applicable,
`worker-stderr.txt`, and a successful call count in `details.json` under Harbor's
verifier logs. Wrong answers, compilation failures, runtime errors, and timeouts
are reported separately. Per-call timeout is 10 seconds; compilation timeout is
120 seconds; Harbor's overall verifier timeout is 300 seconds.

## Qwen through vLLM

The server at `http://127.0.0.1:8000/v1` serves `Qwen/Qwen3.5-9B`.
The default launcher runs question 3274 in all four languages using the generated
`benchmarks/leetcode/qwen-smoke-v2` tasks:

```bash
scripts/eval/leetcode.sh
```

To generate another four-language smoke set and run it:

```bash
.venv/bin/python -m msl_multilingual_self_learning.benchmark.leetcode.unified_tasks \
  --question-id 3243 --output benchmarks/leetcode/my-qwen-smoke
TASKS=benchmarks/leetcode/my-qwen-smoke AGENT=model \
MODEL_BASE_URL=http://127.0.0.1:8000/v1 scripts/eval/leetcode.sh
```

The prompt uses one template: language, native entrypoint/container, the original
problem statement, and identical instructions to return complete source code.
The same generic compilation requirement (including no unused imports) applies to
every language. There is no starter code, solution hint, or hidden-test content. The agent infers
the language from each task, requests source with thinking disabled at temperature
zero (4096-token limit), and writes only `/workspace/solution.json`. The exact
submission is also retained in the Harbor agent logs for review. Truncated model
responses fail explicitly.

`benchmarks/leetcode/qwen-tasks-v2` contains 724 generated model-only tasks. To run
that full set, select it with `TASKS=benchmarks/leetcode/qwen-tasks-v2`; this is a much
larger evaluation than the default smoke test. Use `--language` during generation
to restrict a dataset to one language. `LANGUAGE` on the launcher is optional and
must match every selected task when supplied. Set `N_CONCURRENT`, `JOB_NAME`,
`MODEL`, `MODEL_BASE_URL`, `MODEL_API_KEY`, or `ENV` as needed.

Problem 3319 is always recorded in `exclusions.json` when selected and omitted
from task generation. Other unsupported transports abort generation unless
`--skip-unsupported` is supplied. Generation does not publish partial batches.
Oracle reference files are generated only with `--with-oracle`; model tasks
contain no reference solutions.

## Checks and current scope

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m msl_multilingual_self_learning.benchmark.leetcode.validate_dataset
```

The native transport supports the scalar/array/list types used by the selected
smoke problems. TreeNode/ListNode and in-place/void-return transport are explicitly
rejected; problem 3319 therefore remains outside the executable subset. Dataset
interface validation is not proof that all 182 problems pass reference solutions.
HF tests can also disagree with accepted LeetCode solutions; oracle validation
must precede interpreting model scores. Full 728-task oracle validation remains
a separate step.

`scripts/eval/run.sh` retains the earlier [SWE-bench Multilingual workflow](docs/swebench.md).
