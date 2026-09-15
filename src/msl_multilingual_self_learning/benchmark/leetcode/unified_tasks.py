"""Build Harbor tasks with live native calls and unchanged Python assertions."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from .interfaces.doocs import find_solution_file
from .native_adapters import render_worker

LANGUAGES = ("python", "cpp", "go", "java")
OMITTED_QUESTIONS = {3319: "Omitted pending tree transport support"}
ROOT = Path("benchmarks/leetcode/unified_tasks")
DATASET = Path("benchmarks/leetcode/data/leetcode_multilingual.jsonl")

def format_entrypoint_requirement(language: str, interface: dict[str, Any]) -> str:
    return (
        f"Entrypoint: {interface['raw_signature'].strip()}\n"
        f"Container: {interface.get('container') or '(none)'}"
    )


def build_prompt(record: dict[str, Any], language: str) -> str:
    """One template for all languages; no starter code, examples of code, or tests."""
    interface = record["interfaces"][language]
    source_file = {"python": "solution.py", "cpp": "solution.cpp",
                   "go": "solution.go", "java": "Solution.java"}[language]
    return (
        f"Language: {language}\n"
        f"{format_entrypoint_requirement(language, interface)}\n\n"
        f"Problem:\n{record['problem_description'].strip()}\n\n"
        "Solve the problem in the specified language using the required entrypoint. "
        "Write a complete source file, including any required imports, includes, "
        "and package declarations. Do not define a program entrypoint. "
        "The source must compile or run as submitted. Do not include unused imports. "
        f"Save your final solution to the file /workspace/{source_file} "
        "(the working directory is /workspace). Write ONLY that source file; "
        "do not wrap it in Markdown fences or add explanations. "
        "When the file is written and you are done, issue the command "
        "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`."
    )

# Per-language system requirements, verified against the prebuilt self-contained sif
# (no cross-language apt install). name -> shell checks that must all pass.
SETUP_BY_LANGUAGE = {
    "python": ["python3 --version"],
    "cpp": ["python3 --version", "g++ --version | head -n 1", "[ -f /usr/include/nlohmann/json.hpp ]"],
    "go": ["python3 --version", "go version"],
    "java": ["python3 --version", "javac -version", "[ -f /usr/share/java/gson.jar ]"],
}


def _setup_script(language: str) -> str:
    checks = "\n".join(f"echo '  -- {c}' && {c}" for c in SETUP_BY_LANGUAGE[language])
    return f'''#!/usr/bin/env bash
set -euo pipefail
# Self-contained image: verify only what {language} tasks need; do not apt install.
missing=0
{checks} || missing=$?
if [ "$missing" -ne 0 ]; then
    echo "[setup] {language} toolchain incomplete in image; aborting" >&2
    exit 1
fi
mkdir -p /opt/leetcode
cp /staging/env_files/adapters/* /opt/leetcode/
'''


PYTHON_WORKER = '''import contextlib
import json
import sys
from typing import *
from collections import *
from functools import *
from itertools import *
from heapq import *
from bisect import *
from math import *
import collections, functools, itertools, heapq, bisect, math, random, string
with contextlib.redirect_stdout(sys.stderr):
    exec(open("solution.py").read(), globals())
    candidate = TARGET
for line in sys.stdin:
    args = json.loads(line)
    with contextlib.redirect_stdout(sys.stderr):
        result = candidate(*args)
    print(json.dumps(result, allow_nan=False), flush=True)
'''


SUBMISSION_SOURCES = {"python": "solution.py", "cpp": "solution.cpp", "go": "solution.go", "java": "Solution.java"}


def _submission_packager(language: str, source_file: str) -> str:
    """Deterministic submit hook: package the agent's written source file into the
    solution.json the verifier (judge_runtime.py) reads. Runs before the verifier.
    If a valid solution.json already exists (e.g. oracle), it is kept."""
    return f'''#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("LEETCODE_WORKSPACE", "/workspace"))
LANGUAGE = "{language}"
SOURCE_FILE = "{source_file}"
TARGET = WORKSPACE / "solution.json"

if TARGET.exists():
    try:
        sub = json.loads(TARGET.read_text())
        if str(sub.get("language")) == LANGUAGE and str(sub.get("code", "")).strip():
            print("[package] existing solution.json valid; keeping")
            sys.exit(0)
    except Exception:
        pass

src = WORKSPACE / SOURCE_FILE
if not src.exists():
    print(f"[package] source not found: {{src}}", file=sys.stderr)
    sys.exit(1)
code = src.read_text().strip()
if not code:
    print(f"[package] source empty: {{src}}", file=sys.stderr)
    sys.exit(1)
TARGET.write_text(json.dumps({{"language": LANGUAGE, "code": code}}))
print(f"[package] packaged {{src.name}} -> solution.json ({{len(code)}} chars)")
'''


def canonical_names(problem):
    # New datasets retain the HF argument names explicitly. Older datasets are
    # accepted only when every test keyword agrees with the Doocs Python names.
    names = problem.get("metadata", {}).get("canonical_parameter_names")
    if names is None:
        names = [p["name"] for p in problem["interfaces"]["python"]["parameters"]]
        for node in ast.walk(ast.parse(problem["canonical_tests"]["source"])):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "candidate":
                if any(k.arg not in names for k in node.keywords):
                    raise ValueError("HF/Doocs parameter names differ; regenerate dataset to retain canonical_parameter_names")
    return names


def oracle_code(problem, language):
    path = find_solution_file(int(problem["question_id"]), language)
    if path is None:
        raise ValueError(f"Missing Doocs oracle for {problem['question_id']}/{language}")
    code = path.read_text()
    if language == "go" and not re.search(r"^package\s", code, re.M):
        # Doocs stores LeetCode snippets without a package/import preamble.
        packages = ("sort", "slices", "strings", "strconv", "math", "fmt", "container/heap", "container/list", "math/bits", "unicode")
        imports = [p for p in packages if re.search(r"\b" + p.split('/')[-1] + r"\.", code)]
        code = "package main\n" + "".join(f'import "{p}"\n' for p in imports) + code
    if language == "java":
        code = "import java.util.*;\nimport java.math.*;\n" + code
    return code


def generate(problem, language, output=ROOT, with_oracle=False, images=None):
    # Fail explicitly until object/mutation transport is implemented. Never
    # silently remove cases or coerce native signatures to Python types.
    interface = problem["interfaces"][language]
    all_types = [p["type"] for p in interface["parameters"]] + [interface["return_type"]]
    if any(any(t in value for t in ("TreeNode", "ListNode", "Node")) for value in all_types):
        raise NotImplementedError("Object transport is not supported yet")
    if interface["return_type"] in ("void", "None", "NoneType", ""):
        raise NotImplementedError("In-place/void outputs require mutation transport")
    names = canonical_names(problem)
    if len(names) != len(interface["parameters"]):
        raise ValueError("Canonical/native parameter counts differ")
    code = oracle_code(problem, language) if with_oracle else None
    worker = render_worker(problem, language) if language != "python" else None
    task = output / f"{problem['question_id']}-{language}"
    if task.exists():
        raise FileExistsError(f"Use a fresh output directory: {task}")
    for directory in ("tests", "solution", "environment/files/adapters"):
        (task / directory).mkdir(parents=True)
    (task / "instruction.md").write_text(build_prompt(problem, language))
    config = f'''version = "1.0"
[metadata]
language = "{language}"
question_id = {problem['question_id']}
[agent]
timeout_sec = 300
[verifier]
timeout_sec = 300
[environment]
build_timeout_sec = 900
cpus = 1
memory_mb = 8192
'''
    # docker_image: per-language prebuilt self-contained .sif (absolute path),
    # used directly by harbor's Singularity env (no pull, no Dockerfile build).
    image = (images or {}).get(language)
    if image:
        config += f"docker_image = {json.dumps(str(Path(image).resolve()))}\n"
    (task / "task.toml").write_text(config)
    # Dockerfile is ignored when docker_image (.sif) is set; keep a minimal stub.
    (task / "environment/Dockerfile").write_text("FROM ubuntu:24.04\nWORKDIR /workspace\n")
    (task / "environment/files/setup.sh").write_text(_setup_script(language))
    shutil.copyfile(Path(__file__).with_name("judge_runtime.py"), task / "tests/test.py")
    (task / "tests/_package_submission.py").write_text(_submission_packager(language, SUBMISSION_SOURCES[language]))
    (task / "tests/config.json").write_text(json.dumps({"language": language, "parameter_names": names}))
    (task / "tests/canonical_test.py").write_text(problem["canonical_tests"]["source"])
    (task / "tests/test.sh").write_text('#!/usr/bin/env bash\nset -uo pipefail\n# Submit hook: package agent source -> solution.json.\nif [ -f /tests/_package_submission.py ]; then python3 /tests/_package_submission.py || true; fi\npython3 /tests/test.py\n')
    if language == "python":
        target = (f"{interface['container']}()." if interface.get("container") else "") + interface["callable"]
        (task / "environment/files/adapters/worker.py").write_text(PYTHON_WORKER.replace("TARGET", target))
    else:
        filename = {"cpp": "runner.cpp", "go": "runner.go", "java": "Runner.java"}[language]
        (task / "environment/files/adapters" / filename).write_text(worker)
    if with_oracle:
        (task / "solution/solution.json").write_text(json.dumps({"language": language, "code": code}))
        (task / "solution/solve.sh").write_text('#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p /workspace\ncp /solution/solution.json /workspace/solution.json\n')
        (task / "solution/solve.sh").chmod(0o755)
    (task / "tests/test.sh").chmod(0o755)
    (task / "environment/files/setup.sh").chmod(0o755)
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--question-id", type=int, action="append")
    parser.add_argument("--language", choices=LANGUAGES, action="append")
    parser.add_argument("--with-oracle", action="store_true")
    parser.add_argument("--skip-unsupported", action="store_true", help="Record unsupported transports in exclusions.json instead of aborting")
    parser.add_argument("--image-python", help="Prebuilt local Singularity .sif for python tasks")
    parser.add_argument("--image-cpp", help="Prebuilt local Singularity .sif for cpp tasks")
    parser.add_argument("--image-go", help="Prebuilt local Singularity .sif for go tasks")
    parser.add_argument("--image-java", help="Prebuilt local Singularity .sif for java tasks")
    args = parser.parse_args()
    images = {lang: getattr(args, f"image_{lang}") for lang in LANGUAGES
              if getattr(args, f"image_{lang}")}
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    if args.question_id:
        rows = [r for r in rows if r["question_id"] in args.question_id]
        if {r["question_id"] for r in rows} != set(args.question_id):
            parser.error("Requested question ID is absent from dataset")
    if args.output.exists():
        parser.error(f"Use a fresh output directory: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Stage the complete batch so a rejected problem cannot leave partial tasks.
    with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
        staged = Path(directory) / "tasks"
        staged.mkdir()
        exclusions = []
        for row in rows:
            for language in args.language or LANGUAGES:
                if row["question_id"] in OMITTED_QUESTIONS:
                    exclusions.append({"question_id": row["question_id"], "language": language, "reason": OMITTED_QUESTIONS[row["question_id"]]})
                    continue
                try:
                    generate(row, language, staged, args.with_oracle, images)
                except NotImplementedError as exc:
                    if not args.skip_unsupported:
                        raise
                    exclusions.append({"question_id": row["question_id"], "language": language, "reason": str(exc)})
        if exclusions:
            (staged / "exclusions.json").write_text(json.dumps(exclusions, indent=2) + "\n")
        staged.rename(args.output)
    print(f"Generated {sum(1 for p in args.output.iterdir() if p.is_dir())} tasks in {args.output}; excluded {len(exclusions)}")


if __name__ == "__main__":
    main()