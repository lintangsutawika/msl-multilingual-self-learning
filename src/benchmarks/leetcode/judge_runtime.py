"""Standalone Harbor verifier: one submission and Python check(candidate) for every language."""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import traceback

WORKSPACE = Path(os.environ.get("LEETCODE_WORKSPACE", "/workspace"))
TESTS = Path(__file__).parent
ADAPTERS = Path(os.environ.get("LEETCODE_ADAPTERS", "/opt/leetcode"))
LOGS = Path(os.environ.get("LEETCODE_LOGS", "/logs/verifier"))
FILES = {
    "python": "solution.py",
    "cpp": "solution.cpp",
    "go": "solution.go",
    "java": "Solution.java",
    "rust": "solution.rs",
    "javascript": "solution.js",
    "typescript": "solution.ts",
    "php": "solution.php",
    "ruby": "solution.rb",
}

class CompileError(RuntimeError):
    pass


def prepare(language):
    submission = json.loads((WORKSPACE / "solution.json").read_text())
    if submission.get("language") != language:
        raise ValueError("Submission language does not match task language")
    code = submission.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Submission must contain nonempty source code")
    (WORKSPACE / FILES[language]).write_text(code)
    worker = ADAPTERS / {
        "python": "worker.py",
        "javascript": "runner.js",
        "cpp": "runner.cpp",
        "go": "runner.go",
        "java": "Runner.java",
        "rust": "runner.rs",
        "typescript": "runner.ts",
        "php": "runner.php",
        "ruby": "runner.rb",
    }[language]
    (WORKSPACE / worker.name).write_text(worker.read_text())
    if language == "rust":
        (WORKSPACE / "Cargo.toml").write_text(
            """[package]
    name = "leetcode_runner"
    version = "0.1.0"
    edition = "2021"

    [[bin]]
    name = "leetcode_runner"
    path = "runner.rs"

    [dependencies]
    serde_json = "1"
    """
        )
    if language == "typescript":
        solution = (WORKSPACE / "solution.ts").read_text()
        runner = (WORKSPACE / "runner.ts").read_text()

        combined = WORKSPACE / "combined.ts"
        combined.write_text(
            solution
            + "\n\n"
            + runner
        )
    commands = {
        "cpp": ["g++", "-std=c++17", "-O2", "runner.cpp", "-o", "runner"],
        "go": ["go", "build", "-o", "runner", "solution.go", "runner.go"],
        "java": ["javac", "-cp", "/usr/share/java/gson.jar", "Solution.java", "Runner.java"],
        "rust": [
            "cargo",
            "build",
            "--offline",
            "--release",
        ],
        "typescript": [
            "tsc",
            "--target",
            "ES2020",
            "--module",
            "commonjs",
            "--skipLibCheck",
            "combined.ts",
        ],
    }
    if language in commands:
        result = subprocess.run(commands[language], cwd=WORKSPACE, capture_output=True, text=True, timeout=120)
        (LOGS / "compile.txt").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise CompileError(result.stderr)
    return {
        "python": [sys.executable, "worker.py"],
        "cpp": [str(WORKSPACE / "runner")],
        "go": [str(WORKSPACE / "runner")],
        "java": ["java", "-cp", ".:/usr/share/java/gson.jar", "Runner"],
        "rust": [
            str(
                WORKSPACE
                / "target"
                / "release"
                / "leetcode_runner"
            )
        ],
        "javascript": [
            "node",
            "runner.js",
        ],
        "typescript": [
            "node",
            "combined.js",
        ],
        "php": [
            "php",
            "runner.php",
        ],
        "ruby": [
            "ruby",
            "runner.rb",
        ],
    }[language]


def verify():
    config = json.loads((TESTS / "config.json").read_text())
    signature = inspect.Signature([
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in config["parameter_names"]
    ])
    command = prepare(config["language"])
    with (LOGS / "worker-stderr.txt").open("w") as errors:
        process = subprocess.Popen(command, cwd=WORKSPACE, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        pending = bytearray()
        calls = 0

        def candidate(*args, **kwargs):
            nonlocal calls
            bound = signature.bind(*args, **kwargs)
            ordered = [bound.arguments[name] for name in config["parameter_names"]]
            process.stdin.write((json.dumps(ordered, allow_nan=False) + "\n").encode())
            process.stdin.flush()
            while b"\n" not in pending:
                if not selector.select(timeout=10):
                    raise TimeoutError("Native candidate exceeded 10 seconds per call")
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("Candidate exited without a JSON result; see worker-stderr.txt")
                pending.extend(chunk)
                if len(pending) > 64 * 1024 * 1024:
                    raise RuntimeError("Candidate output exceeds 64 MiB")
            line, _, rest = pending.partition(b"\n")
            pending[:] = rest
            calls += 1
            return json.loads(line)

        try:
            namespace = {}
            exec("from typing import *\nfrom math import *\nfrom collections import *\nfrom functools import *\nfrom itertools import *\nimport math, collections, functools, itertools, random, string\n", namespace)
            exec((TESTS / "canonical_test.py").read_text(), namespace)
            namespace["check"](candidate)
            if not calls:
                raise RuntimeError("Canonical test made no candidate calls")
            process.stdin.close()
            process.wait(timeout=10)
            extra = bytes(pending) + process.stdout.read()
            if process.returncode or extra.strip():
                raise RuntimeError("Candidate failed at shutdown or emitted extra output")
            (LOGS / "details.json").write_text(json.dumps({"language": config["language"], "calls": calls}))
        finally:
            selector.close()
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
            if not process.stdin.closed:
                process.stdin.close()


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "reward.txt").write_text("0\n")
    status = "RUNTIME_ERROR"
    try:
        verify()
        status = "PASS"
    except CompileError:
        status = "COMPILE_ERROR"
        traceback.print_exc()
    except AssertionError:
        status = "WRONG_ANSWER"
        traceback.print_exc()
    except (TimeoutError, subprocess.TimeoutExpired):
        status = "TIMEOUT"
        traceback.print_exc()
    except Exception:
        traceback.print_exc()
    (LOGS / "status.txt").write_text(status + "\n")
    (LOGS / "reward.txt").write_text("1\n" if status == "PASS" else "0\n")
    print(status)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
