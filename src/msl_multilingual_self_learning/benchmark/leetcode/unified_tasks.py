"""Build Harbor tasks with live native calls and unchanged Python assertions."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import shutil
import tempfile

from .interfaces.doocs import find_solution_file
from .native_adapters import render_worker
from .run_one import build_prompt

LANGUAGES = ("python", "cpp", "go", "java")
OMITTED_QUESTIONS = {3319: "Omitted pending tree transport support"}
ROOT = Path("benchmarks/leetcode/unified_tasks")
DATASET = Path("benchmarks/leetcode/data/leetcode_multilingual.jsonl")
PACKAGES = "python3 python3-venv python3-pip g++ golang-go default-jdk-headless nlohmann-json3-dev libgoogle-gson-java tmux asciinema ca-certificates"
SETUP = f'''#!/usr/bin/env bash
set -euo pipefail
if ! command -v g++ >/dev/null || ! command -v go >/dev/null || ! command -v javac >/dev/null || [ ! -f /usr/share/java/gson.jar ] || [ ! -f /usr/include/nlohmann/json.hpp ]; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends {PACKAGES}
fi
python3 --version
g++ --version | head -n 1
go version
javac -version
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
        packages = ("sort", "slices", "strings", "strconv", "math", "fmt", "container/heap", "container/list", "math/bits")
        imports = [p for p in packages if re.search(r"\b" + p.split('/')[-1] + r"\.", code)]
        code = "package main\n" + "".join(f'import "{p}"\n' for p in imports) + code
    if language == "java":
        code = "import java.util.*;\nimport java.math.*;\n" + code
    return code


def generate(problem, language, output=ROOT, with_oracle=False, image=None):
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
memory_mb = 4096
'''
    if image:
        config += f"docker_image = {json.dumps(str(Path(image).resolve()))}\n"
    (task / "task.toml").write_text(config)
    (task / "environment/Dockerfile").write_text(f"FROM ubuntu:24.04\nRUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends {PACKAGES}\nCOPY files/adapters/ /opt/leetcode/\nWORKDIR /workspace\n")
    (task / "environment/files/setup.sh").write_text(SETUP)
    shutil.copyfile(Path(__file__).with_name("judge_runtime.py"), task / "tests/test.py")
    (task / "tests/config.json").write_text(json.dumps({"language": language, "parameter_names": names}))
    (task / "tests/canonical_test.py").write_text(problem["canonical_tests"]["source"])
    (task / "tests/test.sh").write_text('#!/usr/bin/env bash\nset -uo pipefail\npython3 /tests/test.py\n')
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
    parser.add_argument("--image", help="Prebuilt local Singularity .sif runtime")
    args = parser.parse_args()
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
                    generate(row, language, staged, args.with_oracle, args.image)
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
