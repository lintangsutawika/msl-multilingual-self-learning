"""Standalone Harbor verifier: one submission and Python check(candidate) for every language."""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import subprocess
import sys
import traceback

WORKSPACE = Path(os.environ.get("LEETCODE_WORKSPACE", "/workspace"))
TESTS = Path(__file__).parent
ADAPTERS = Path(os.environ.get("LEETCODE_ADAPTERS", "/opt/leetcode"))
LOGS = Path(os.environ.get("LEETCODE_LOGS", "/logs/verifier"))
# Python solutions run on the image's interpreter. Harbor's exec server puts /usr/bin
# first on PATH, where the Debian python3 pulled in by Harbor's own tools lives.
SOLUTION_PYTHON = "/usr/local/bin/python3" if Path("/usr/local/bin/python3").exists() else sys.executable
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
WORKERS = {
    "python": "worker.py",
    "javascript": "runner.js",
    "cpp": "runner.cpp",
    "go": "runner.go",
    "java": "Runner.java",
    "rust": "runner.rs",
    "typescript": "runner.ts",
    "php": "runner.php",
    "ruby": "runner.rb",
}

# Imports LeetCode makes available without the solution asking for them.
JAVA_IMPORTS = (
    "import java.util.*;\n"
    "import java.util.function.*;\n"
    "import java.util.stream.*;\n"
    "import java.math.*;\n"
)
# Standard Go packages the judge will import when the solution uses them
# without an import (identifier -> import path).
GO_PACKAGES = {
    "bits": "math/bits",
    "big": "math/big",
    "bytes": "bytes",
    "cmp": "cmp",
    "errors": "errors",
    "fmt": "fmt",
    "heap": "container/heap",
    "list": "container/list",
    "maps": "maps",
    "math": "math",
    "rand": "math/rand",
    "regexp": "regexp",
    "slices": "slices",
    "sort": "sort",
    "strconv": "strconv",
    "strings": "strings",
    "unicode": "unicode",
    "utf8": "unicode/utf8",
}
TS_PRELUDE = (
    "for (const leetcodeLevel of [\"log\", \"info\", \"debug\"]) "
    "(console as any)[leetcodeLevel] = console.error;\n"
    # Libraries LeetCode provides as globals (PriorityQueue, Queue, Deque, _).
    "for (const leetcodeName of [\"@datastructures-js/priority-queue\", \"@datastructures-js/queue\", "
    "\"@datastructures-js/deque\", \"lodash\"]) {\n"
    "    let leetcodeLib: any = {};\n"
    "    try { leetcodeLib = require(leetcodeName); } catch (error) {\n"
    "        try { leetcodeLib = require(\"/usr/local/lib/node_modules/\" + leetcodeName); } catch (error2) {}\n"
    "    }\n"
    "    Object.assign(globalThis, leetcodeName === \"lodash\" ? { _: leetcodeLib } : leetcodeLib);\n"
    "}\n"
)
# Types for those globals, declared only where the solution does not define
# the name itself (its own `class Queue` must not clash).
TS_LIBRARY_DECLARATIONS = {
    name: f"declare class {name}<T = any> {{ constructor(...args: any[]); [key: string]: any; static [key: string]: any; }}\n"
    for name in ("PriorityQueue", "MinPriorityQueue", "MaxPriorityQueue", "Queue", "Deque")
}
TS_LIBRARY_DECLARATIONS["_"] = "declare const _: any;\n"


def ts_library_declarations(code):
    return "".join(
        declaration for name, declaration in TS_LIBRARY_DECLARATIONS.items()
        if not re.search(
            r"\b(?:class|function|const|let|var|interface|type|enum)\s+" + re.escape(name) + r"(?![\w$])"
            r"|\bimport\b[^;]*(?<![\w$])" + re.escape(name) + r"(?![\w$])",
            code,
        )
    )
CARGO_TOML = """[package]
name = "leetcode_runner"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "leetcode_runner"
path = "runner.rs"

[dependencies]
serde_json = "1"

# Its own workspace root, so a Cargo.toml the agent left in /workspace is ignored.
[workspace]
"""
DEBIAN_CARGO_REGISTRY = Path("/usr/share/cargo/registry")
# Compiler options are pinned here rather than left to tsc defaults (TS 7 turns
# strict on), so an image rebuild cannot change what counts as a compile error.
# Non-strict: types LeetCode's JavaScript twin never checks do not fail TS.
# ES2022/ES2023 lib: TS may use every built-in plain JS gets on Node 22.
TSCONFIG = {
    "compilerOptions": {
        "target": "ES2022",
        "lib": ["ES2023", "DOM"],
        "module": "commonjs",
        "strict": False,
        "skipLibCheck": True,
    },
    "files": ["combined.ts"],
}


class CompileError(RuntimeError):
    pass


def normalize(language, code, config):
    """Adapt a submission to the runner contract, LeetCode-style.

    Models are judged on their algorithm, not on guessing harness details, so
    any reasonable file structure is accepted: whatever package clause, a
    missing PHP open tag, a user-defined main, or a free function where the
    runner expects a Solution method. Returns (code, notes).
    """
    notes = []
    call = config.get("callable")
    container = config.get("container")
    if language == "go":
        package = re.search(r"^package\s+(\w+)", code, re.M)
        if package is None:
            code = "package main\n" + code
            notes.append("go: added package clause")
        elif package.group(1) != "main":
            code = code[:package.start(1)] + "main" + code[package.end(1):]
            notes.append("go: renamed package to main")
        code, n = re.subn(r"^func\s+main\s*\(\s*\)", "func leetcodeUserMain()", code, flags=re.M)
        if n:
            notes.append("go: renamed user main")
    elif language == "rust":
        code, n = re.subn(r"\bfn\s+main\s*\(", "fn leetcode_user_main(", code)
        if n:
            notes.append("rust: renamed user main")
    elif language == "java":
        code, n = re.subn(r"^\s*package\s+[\w.]+\s*;", "", code, count=1, flags=re.M)
        if n:
            notes.append("java: removed package declaration")
        code = JAVA_IMPORTS + code
    elif language == "php":
        if not code.lstrip().startswith("<?"):
            code = "<?php\n" + code
            notes.append("php: added <?php open tag")
    elif language == "cpp" and container and call:
        if not re.search(r"\b(class|struct)\s+" + container + r"\b", code) and re.search(r"\b" + call + r"\s*\(", code):
            code += (
                f"\n\nclass {container} {{\npublic:\n"
                f"    template <class... A> auto {call}(A&&... a) {{ return ::{call}(std::forward<A>(a)...); }}\n"
                "};\n"
            )
            notes.append("cpp: wrapped free function in Solution")
    return code, notes


def adapt_runner(language, runner, code, config, notes):
    """Adjust the runner to what the (normalized) solution defines."""
    call = config.get("callable")
    container = config.get("container")
    if language == "rust" and container:
        if re.search(r"\bstruct\s+" + container + r"\b", code):
            runner = runner.replace(f"struct {container};\n", "", 1)
            notes.append("rust: solution defines its own struct Solution")
        if call and not re.search(r"\bimpl\s+" + container + r"\b", code) and re.search(r"\bfn\s+" + call + r"\b", code):
            runner = runner.replace(f"{container}::{call}(", f"{call}(")
            notes.append("rust: calling free function")
    return runner


def _fix_go_imports(source, errors, notes):
    """Drop unused imports and add missing standard ones, as reported by go build."""
    changed = False
    for path in re.findall(r'"([\w/]+)" imported (?:as \w+ )?and not used', errors):
        source, n = re.subn(r'^[ \t]*(?:import[ \t]+)?(?:\w+[ \t]+)?"' + re.escape(path) + r'"[ \t]*\n', "", source, flags=re.M)
        if n:
            changed = True
            notes.append(f"go: removed unused import {path}")
    for name in sorted(set(re.findall(r"undefined: (\w+)", errors))):
        if name in GO_PACKAGES:
            source = re.sub(r"^package main\b.*$", lambda m: m.group(0) + f'\nimport "{GO_PACKAGES[name]}"', source, count=1, flags=re.M)
            changed = True
            notes.append(f"go: added import {GO_PACKAGES[name]}")
    return source, changed


def compile_source(language, build, notes):
    commands = {
        "cpp": ["g++", "-std=c++20", "-O2", "runner.cpp", "-o", "runner"],
        "go": ["go", "build", "-o", "runner", "solution.go", "runner.go"],
        "java": ["javac", "-cp", "/usr/share/java/gson.jar", "Solution.java", "Runner.java"],
        "rust": ["cargo", "build", "--offline", "--release"],
        "typescript": ["tsc", "-p", "tsconfig.json"],
    }
    if language not in commands:
        return
    log = []
    # Go refuses unused imports and has no implicit ones; retry after fixing them.
    for _ in range(4 if language == "go" else 1):
        result = subprocess.run(commands[language], cwd=build, capture_output=True, text=True, timeout=120)
        log.append(result.stdout + result.stderr)
        if not result.returncode or language != "go":
            break
        source, changed = _fix_go_imports((build / "solution.go").read_text(), result.stderr, notes)
        if not changed:
            break
        (build / "solution.go").write_text(source)
    (LOGS / "compile.txt").write_text("\n".join(log))
    if result.returncode:
        raise CompileError(result.stderr)


def _load_source(language):
    """Return the agent's source code for `language`, packaging it into
    solution.json when absent (was tests/_package_submission.py)."""
    target = WORKSPACE / "solution.json"
    if target.exists():
        try:
            sub = json.loads(target.read_text())
            if sub.get("language") == language and isinstance(sub.get("code"), str) and sub["code"].strip():
                return sub["code"]
        except Exception:
            pass
    src = WORKSPACE / FILES[language]
    if not src.exists():
        raise ValueError(f"source not found: {src}")
    code = src.read_text().strip()
    if not code:
        raise ValueError(f"source empty: {src}")
    target.write_text(json.dumps({"language": language, "code": code}))
    return code


def prepare(language, config, build):
    code = _load_source(language)
    # Build in a fresh directory so files the agent left in /workspace
    # (tsconfig.json, Cargo.toml, stray sources) cannot affect the verdict.
    code, notes = normalize(language, code, config)
    worker = WORKERS[language]
    runner = adapt_runner(language, (ADAPTERS / worker).read_text(), code, config, notes)
    (build / FILES[language]).write_text(code)
    (build / worker).write_text(runner)
    if language == "rust":
        (build / "Cargo.toml").write_text(CARGO_TOML)
        if DEBIAN_CARGO_REGISTRY.is_dir():
            (build / ".cargo").mkdir()
            (build / ".cargo/config.toml").write_text(
                '[source.crates-io]\nreplace-with = "debian"\n\n'
                f'[source.debian]\ndirectory = "{DEBIAN_CARGO_REGISTRY}"\n'
            )
    if language in ("javascript", "typescript"):
        # Shadow any package.json in /workspace (e.g. "type": "module").
        (build / "package.json").write_text('{"type": "commonjs"}\n')
    if language == "typescript":
        (build / "combined.ts").write_text(TS_PRELUDE + ts_library_declarations(code) + code + "\n\n" + runner)
        (build / "tsconfig.json").write_text(json.dumps(TSCONFIG))
    try:
        compile_source(language, build, notes)
    finally:
        (LOGS / "normalizations.json").write_text(json.dumps(notes))
    return {
        "python": [SOLUTION_PYTHON, "worker.py"],
        "cpp": [str(build / "runner")],
        "go": [str(build / "runner")],
        "java": ["java", "-cp", ".:/usr/share/java/gson.jar", "Runner"],
        "rust": [str(build / "target" / "release" / "leetcode_runner")],
        "javascript": ["node", "runner.js"],
        "typescript": ["node", "combined.js"],
        "php": ["php", "runner.php"],
        "ruby": ["ruby", "runner.rb"],
    }[language]


def verify():
    config = json.loads((TESTS / "config.json").read_text())
    signature = inspect.Signature([
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in config["parameter_names"]
    ])
    # A fresh directory on the workspace disk (which has room for Rust/Go
    # builds, unlike a possibly tmpfs-backed /tmp), rebuilt for every run.
    build = WORKSPACE / ".leetcode-build"
    shutil.rmtree(build, ignore_errors=True)
    build.mkdir()
    try:
        run_candidate(config, signature, build)
    finally:
        shutil.rmtree(build, ignore_errors=True)


def run_candidate(config, signature, build):
    command = prepare(config["language"], config, build)
    with (LOGS / "worker-stderr.txt").open("w") as errors:
        process = subprocess.Popen(command, cwd=build, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors)
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
