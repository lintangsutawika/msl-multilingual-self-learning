"""Adapter: generate Harbor leetcode tasks from per-language templates.

For each (problem, language) the per-language template directory
(task-template-<lang>) is copied, its {{ }} placeholders are substituted with
problem-specific values, and the problem-specific verifier artifacts (judge
runtime, canonical tests, parameter config, native runner/worker) are rendered
into the task. Mirrors material-harbor/matqna's adapter pattern.
"""
from __future__ import annotations

import ast
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .native_adapters import render_worker

LANGUAGES = (
    "python",
    "cpp",
    "go",
    "java",
    "rust",
    "javascript",
    "typescript",
    "php",
    "ruby",
)
OMITTED_QUESTIONS = {3319: "Omitted pending tree transport support"}
SOURCE_FILES = {
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
ADAPTERS = {
    "cpp": "runner.cpp",
    "go": "runner.go",
    "java": "Runner.java",
    "rust": "runner.rs",
    "javascript": "runner.js",
    "php": "runner.php",
    "typescript": "runner.ts",
    "ruby": "runner.rb",
}

PKG = Path(__file__).parent


def canonical_names(problem: dict[str, Any]) -> list[str]:
    """Canonical parameter names for the shared Python judge."""
    names = problem.get("metadata", {}).get("canonical_parameter_names")
    if names is None:
        names = [p["name"] for p in problem["interfaces"]["python"]["parameters"]]
        for node in ast.walk(ast.parse(problem["canonical_tests"]["source"])):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "candidate":
                if any(k.arg not in names for k in node.keywords):
                    raise ValueError(
                        "HF/Doocs parameter names differ; regenerate dataset to retain canonical_parameter_names"
                    )
    return names


# Constraint audit of the canonical tests (see constraints.py).
INVALID_TESTS = Path("benchmarks/leetcode/data/invalid_tests.json")
# A problem needs this many fair test cases, or a wrong solution can pass by luck.
MIN_VALID_TESTS = 10

INT32_MAX = 2**31 - 1
INT64_MAX = 2**63 - 1
JS_SAFE_MAX = 2**53 - 1


def integer_limit(language: str, type_name: str) -> int | None:
    """Largest integer a declared LeetCode type holds exactly (None: unbounded or not an integer)."""
    if language in ("python", "ruby"):
        return None
    if language in ("javascript", "typescript"):
        return JS_SAFE_MAX
    if language == "php":
        return INT64_MAX
    if language == "go":
        return INT64_MAX if re.search(r"\bint(64)?\b", type_name) else None
    if language == "rust":
        return INT64_MAX if "i64" in type_name else INT32_MAX if "i32" in type_name else None
    if re.search(r"\blong\b", type_name):
        return INT64_MAX
    if re.search(r"\b(int|Integer)\b", type_name):
        return INT32_MAX
    return None


def _integers(value: Any):
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _integers(item)


def _non_finite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    return isinstance(value, (list, tuple)) and any(_non_finite(item) for item in value)


def _unrepresentable_languages(problem: dict[str, Any], args: list[Any], expected: Any) -> list[str]:
    """Languages whose declared parameter/return types cannot hold this test case."""
    if _non_finite(args) or _non_finite(expected):
        # inf/nan cannot cross the JSON transport, so no language can pass.
        return list(problem["interfaces"])
    languages = []
    for language, interface in problem["interfaces"].items():
        pairs = list(zip((p["type"] for p in interface["parameters"]), args))
        pairs.append((interface["return_type"], expected))
        for type_name, value in pairs:
            limit = integer_limit(language, type_name)
            if limit is not None and any(not -limit - 1 <= x <= limit for x in _integers(value)):
                languages.append(language)
                break
    return languages


_ARITHMETIC_NODES = (
    ast.Expression, ast.Constant, ast.List, ast.Tuple, ast.Load,
    ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Pow, ast.USub, ast.UAdd,
)
# Names the reference solution's outputs may contain (the judge star-imports math).
_TEST_NAMES = {"inf": math.inf, "nan": math.nan}


def _test_value(node: ast.AST) -> Any:
    """Value of a test argument: a literal, or plain arithmetic such as 10**9, [0] * n or -inf."""
    try:
        return ast.literal_eval(node)
    except ValueError:
        expression = ast.Expression(node)
        for n in ast.walk(expression):
            if not isinstance(n, _ARITHMETIC_NODES) and not (isinstance(n, ast.Name) and n.id in _TEST_NAMES):
                raise
        return eval(compile(expression, "<test>", "eval"), {"__builtins__": {}, **_TEST_NAMES})


def filter_canonical_tests(
    problem: dict[str, Any],
    invalid: dict[str, list[str]] | None = None,
) -> tuple[str, list[dict[str, Any]], int]:
    """Drop test cases that are not fair for every language.

    LeetCodeDataset generated its test inputs and recorded the Python
    reference's output, so some cases break the problem's own contract:

    * values that do not fit a declared LeetCode type (board values of 7e9
      where LeetCode promises |x| <= 1e9 cannot even be received in
      Rust/C++/Java's `int`), and expected outputs of inf/-inf (the reference's
      sentinel), which no language can return;
    * inputs that violate the problem's Constraints, listed in `invalid`
      (assert source -> violated rules, from constraints.py's audit).

    Such cases are removed for every language. Returns (source, dropped,
    number of test cases kept).
    """
    invalid = invalid or {}
    source = problem["canonical_tests"]["source"]
    names = canonical_names(problem)
    asserts = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assert)
        and isinstance(node.test, ast.Compare)
        and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)
        and isinstance(node.test.left, ast.Call)
        and isinstance(node.test.left.func, ast.Name) and node.test.left.func.id == "candidate"
    ]
    dropped = []
    for node in asserts:
        test = ast.unparse(node)
        if test in invalid:
            dropped.append({"line": node.lineno, "end_line": node.end_lineno, "violates": invalid[test]})
            continue
        call = node.test.left
        try:
            values = [_test_value(a) for a in call.args]
            keywords = {k.arg: _test_value(k.value) for k in call.keywords}
            expected = _test_value(node.test.comparators[0])
        except (ValueError, TypeError, SyntaxError, RecursionError):
            continue
        values += [keywords[n] for n in names[len(values):] if n in keywords]
        if len(values) != len(names):
            continue
        languages = _unrepresentable_languages(problem, values, expected)
        if languages:
            dropped.append({"line": node.lineno, "end_line": node.end_lineno, "unrepresentable_in": languages})
    kept = len(asserts) - len(dropped)
    if not dropped:
        return source, [], kept
    removed = {n for d in dropped for n in range(d["line"], d["end_line"] + 1)}
    lines = source.splitlines(keepends=True)
    return "".join(line for n, line in enumerate(lines, 1) if n not in removed), dropped, kept


def load_invalid_tests(path: Path = INVALID_TESTS) -> dict[int, dict[str, list[str]]]:
    """question_id -> {assert source: violated rules} from constraints.py's audit."""
    if not path.is_file():
        return {}
    return {
        int(entry["question_id"]): {t["test"]: t["violates"] for t in entry["invalid_tests"]}
        for entry in json.loads(path.read_text())
    }


def _fill(path: Path, **kw: str) -> None:
    text = path.read_text()
    for k, v in kw.items():
        text = text.replace("{" + k + "}", str(v))
    path.write_text(text)





def dockerfile_to_def(dockerfile: Path) -> str | None:
    """Translate a Dockerfile's dependency-bearing instructions (RUN/COPY/ENV)
    into a Singularity recipe fragment (%post/%files/%environment) for layering
    onto a base image. Returns None when the Dockerfile has no such layers."""
    if not dockerfile.exists():
        return None
    posts: list[str] = []
    files: list[str] = []
    envs: list[str] = []
    current: list[str] | None = None

    def flush_run() -> None:
        nonlocal current
        if current:
            block = " ".join(ln.strip() for ln in current).strip()
            if block:
                posts.append(block)
            current = None

    try:
        lines = dockerfile.read_text().splitlines()
    except OSError:
        return None

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if current is not None:
            if raw.rstrip().endswith("\\"):
                current.append(raw[: raw.rindex("\\")].strip())
                i += 1
                continue
            current.append(raw.strip())
            flush_run()
            i += 1
            continue

        if not line or line.startswith("#"):
            i += 1
            continue
        upper = line.upper()
        if upper.startswith("RUN "):
            flush_run()
            body = line[4:].strip()
            if raw.rstrip().endswith("\\"):
                current = [body[: body.rindex("\\")].strip()]
            else:
                current = None
                posts.append(body)
            i += 1
            continue
        if upper.startswith("COPY "):
            flush_run()
            parts = line[5:].split()
            if len(parts) >= 2:
                srcs, dst = parts[:-1], parts[-1]
                for src in srcs:
                    files.append(f"    {src} {dst}")
                posts.append(f"mkdir -p {dst}")
            i += 1
            continue
        if upper.startswith("ENV "):
            flush_run()
            body = line[4:].strip()
            if "=" in body:
                k, _, v = body.partition("=")
                envs.append(f"    export {k.strip()}={v.strip()}")
            else:
                bits = body.split(None, 1)
                if len(bits) == 2:
                    envs.append(f"    export {bits[0]}={bits[1]}")
            i += 1
            continue
        flush_run()
        i += 1

    flush_run()
    if not (posts or files or envs):
        return None

    block = []
    if files:
        block.append("%files")
        block.extend(files)
    if posts:
        block.append("%post")
        block.append("    set -e")
        block.extend(f"    {p}" for p in posts)
    if envs:
        block.append("%environment")
        block.extend(envs)
    return "\n".join(block) + "\n"


def prebuild_sif(dockerfile: Path, out_sif: Path, *, container_bin: str = "singularity") -> Path:
    """Build a Singularity sif from a task Dockerfile (FROM + RUN/COPY/ENV deps).

    Writes a .def alongside out_sif, runs ``singularity build --fakeroot`` with the
    Dockerfile's dir as the build context (so COPY sources resolve), and returns
    out_sif. Raises CalledProcessError on a failed build.
    """
    dockerfile = Path(dockerfile).resolve()
    out_sif = Path(out_sif).resolve()
    out_sif.parent.mkdir(parents=True, exist_ok=True)
    base = dockerfile_to_def(dockerfile)
    # Base image = the Dockerfile's FROM (last non-comment FROM line).
    base_image = None
    try:
        for raw in dockerfile.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper().startswith("FROM "):
                ref = line.split(None, 1)[1].strip()
                if " AS " in ref.upper():
                    ref = ref[: ref.upper().index(" AS ")].strip()
                base_image = ref
    except OSError:
        base_image = None
    if not base_image:
        raise ValueError(f"No FROM image in {dockerfile}")

    fragment = base  # may be None for FROM/WORKDIR-only; fine
    def_path = out_sif.with_name(out_sif.name + ".def")
    def_lines = ["Bootstrap: docker", f"From: {base_image}", ""]
    if fragment:
        def_lines.append(fragment)
    def_path.write_text("\n".join(def_lines))

    subprocess.run(
        [container_bin, "build", "--fakeroot", str(out_sif), str(def_path)],
        check=True,
        cwd=str(dockerfile.parent),
    )
    return out_sif


def prebuild_language_sifs(
    languages: tuple[str, ...],
    out_dir: Path,
    *,
    template_dir: Path | None = None,
    container_bin: str = "singularity",
) -> dict[str, str]:
    """Build one sif per language from its template Dockerfile, into out_dir.

    Returns a dict {language: <sif path>} suitable for adapter.generate_all's
    images arg (which becomes task.toml [environment].docker_image)."""
    template_dir = template_dir or PKG
    images: dict[str, str] = {}
    for lang in languages:
        dockerfile = template_dir / f"task-template-{lang}/environment/Dockerfile"
        if not dockerfile.exists():
            raise FileNotFoundError(f"No template Dockerfile for {lang}: {dockerfile}")
        sif = Path(out_dir) / f"{lang}.sif"
        prebuild_sif(dockerfile, sif, container_bin=container_bin)
        images[lang] = str(sif)
        print(f"prebuilt {sif}")
    return images

def generate(
    problem: dict[str, Any],
    language: str,
    output: Path,
    docker_image: str | None = None,
    with_oracle: bool = False,
) -> Path:
    """Render one task for (problem, language) into a fresh subdir of output."""
    interface = problem["interfaces"][language]
    all_types = [p["type"] for p in interface["parameters"]] + [interface["return_type"]]
    if any(any(t in value for t in ("TreeNode", "ListNode", "Node")) for value in all_types):
        raise NotImplementedError("Object transport is not supported yet")
    if interface["return_type"] in ("void", "None", "NoneType", ""):
        raise NotImplementedError("In-place/void outputs require mutation transport")

    names = canonical_names(problem)
    if len(names) != len(interface["parameters"]):
        raise ValueError("Canonical/native parameter counts differ")

    task = output / f"{problem['question_id']}-{language}"
    if task.exists():
        raise FileExistsError(f"Use a fresh output directory: {task}")

    # Copy the per-language template into the new task.
    tpl = PKG / f"task-template-{language}"
    shutil.copytree(tpl, task, dirs_exist_ok=False)

    # When a prebuilt sif supplies docker_image (e.g. --prebuild-sif), the image
    # already carries the full toolchain, so the task's environment/Dockerfile
    # must NOT re-trigger the runtime Dockerfile-dep-layering (which would build a
    # redundant derived sif). Replace it with a minimal no-RUN recipe that keeps
    # WORKDIR so the agent still lands in /workspace.
    if docker_image:
        dockerfile = task / "environment" / "Dockerfile"
        if dockerfile.exists():
            from_line = "ubuntu:24.04"
            try:
                for raw in dockerfile.read_text().splitlines():
                    line = raw.strip()
                    if line.upper().startswith("FROM "):
                        from_line = line.split(None, 1)[1].strip().split()[0]
            except OSError:
                pass
            dockerfile.write_text(
                f"FROM {from_line}\n"
                "WORKDIR /workspace\n"
            )

    # Fill static-template placeholders.
    container = interface.get("container") or "(none)"
    source_file = SOURCE_FILES[language]
    fills = {
        "question_id": problem["question_id"],
        "language": language,
        "docker_image": str(Path(docker_image).resolve()) if docker_image else "",
        "entrypoint": interface["raw_signature"].strip(),
        "container": container,
        "problem": problem["problem_description"].strip(),
        "source_file": source_file,
    }
    _fill(task / "task.toml", **fills)
    _fill(task / "instruction.md", **fills)

    # Problem-specific verifier artifacts.
    (task / "tests/test.py").write_text((PKG / "judge_runtime.py").read_text())
    (task / "tests/config.json").write_text(json.dumps({
        "language": language,
        "parameter_names": names,
        "callable": interface["callable"],
        "container": interface.get("container"),
    }))
    (task / "tests/canonical_test.py").write_text(problem["canonical_tests"]["source"])

    # Native runner/worker.
    if language == "python":
        target = (f"{interface['container']}()." if interface.get("container") else "") + interface["callable"]
        worker_tpl = (PKG / "task-template-python/environment/files/adapters/worker.tpl").read_text()
        (task / "environment/files/adapters/worker.py").write_text(worker_tpl.replace("TARGET", target))
    else:
        adapter_dir = (
            task
            / "environment"
            / "files"
            / "adapters"
        )

        adapter_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        filename = ADAPTERS[language]

        (adapter_dir / filename).write_text(
            render_worker(
                problem,
                language,
            )
        )
    if with_oracle:
        from .interfaces.doocs import find_solution_file
        import re
        path = find_solution_file(int(problem["question_id"]), language)
        if path is None:
            raise ValueError(f"Missing Doocs oracle for {problem['question_id']}/{language}")
        code = path.read_text()
        if language == "go" and not re.search(r"^package\s", code, re.M):
            packages = ("sort", "slices", "strings", "strconv", "math", "fmt", "container/heap", "container/list", "unicode")
            imports = [p for p in packages if re.search(r"\b" + p.split('/')[-1] + r"\.", code)]
            code = "package main\n" + "".join(f'import "{p}"\n' for p in imports) + code
        if language == "java":
            code = "import java.util.*;\nimport java.math.*;\n" + code
        (task / "solution/solution.json").write_text(json.dumps({"language": language, "code": code}))

    return task


def generate_all(
    problems: list[dict[str, Any]],
    output: Path,
    languages: tuple[str, ...] = LANGUAGES,
    images: dict[str, str] | None = None,
    with_oracle: bool = False,
    skip_unsupported: bool = False,
    invalid_tests: dict[int, dict[str, list[str]]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Generate tasks for every (problem, language), staging atomically.

    Returns (exclusions, count). Aborts (raising) on unsupported transports
    unless skip_unsupported, in which case they are recorded as exclusions.
    """
    images = images or {}
    output.parent.mkdir(parents=True, exist_ok=True)
    exclusions: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / "tasks"
        staged.mkdir()
        dropped_tests: list[dict[str, Any]] = []
        for row in problems:
            source, dropped, kept = filter_canonical_tests(row, (invalid_tests or {}).get(int(row["question_id"])))
            if dropped:
                row = {**row, "canonical_tests": {**row["canonical_tests"], "source": source}}
                dropped_tests.append({"question_id": row["question_id"], "kept": kept, "dropped": dropped})
            if kept < MIN_VALID_TESTS and row["question_id"] not in OMITTED_QUESTIONS:
                for language in languages:
                    exclusions.append({"question_id": row["question_id"], "language": language,
                                       "reason": f"Only {kept} valid test cases (< {MIN_VALID_TESTS})"})
                continue
            for language in languages:
                if row["question_id"] in OMITTED_QUESTIONS:
                    exclusions.append({"question_id": row["question_id"], "language": language,
                                       "reason": OMITTED_QUESTIONS[row["question_id"]]})
                    continue
                try:
                    generate(row, language, staged, images.get(language), with_oracle)
                # NotImplementedError: unsupported transport; ValueError: the dataset's
                # canonical tests and native interface disagree (e.g. parameter counts).
                except (NotImplementedError, ValueError) as exc:
                    if not skip_unsupported:
                        raise
                    exclusions.append({"question_id": row["question_id"], "language": language, "reason": str(exc)})
        if exclusions:
            (staged / "exclusions.json").write_text(json.dumps(exclusions, indent=2) + "\n")
        if dropped_tests:
            (staged / "dropped_tests.json").write_text(json.dumps(dropped_tests, indent=2) + "\n")
        staged.rename(output)
    return exclusions, sum(1 for p in output.iterdir() if p.is_dir())