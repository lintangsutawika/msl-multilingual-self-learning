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
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .native_adapters import render_worker

LANGUAGES = ("python", "cpp", "go", "java")
OMITTED_QUESTIONS = {3319: "Omitted pending tree transport support"}
SOURCE_FILES = {
    "python": "solution.py",
    "cpp": "solution.cpp",
    "go": "solution.go",
    "java": "Solution.java",
}
ADAPTERS = {
    "cpp": "runner.cpp",
    "go": "runner.go",
    "java": "Runner.java",
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


def _fill(path: Path, **kw: str) -> None:
    text = path.read_text()
    for k, v in kw.items():
        text = text.replace("{" + k + "}", str(v))
    path.write_text(text)


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
    (task / "tests/config.json").write_text(json.dumps({"language": language, "parameter_names": names}))
    (task / "tests/canonical_test.py").write_text(problem["canonical_tests"]["source"])

    # Native runner/worker.
    if language == "python":
        target = (f"{interface['container']}()." if interface.get("container") else "") + interface["callable"]
        worker_tpl = (PKG / "task-template-python/environment/files/adapters/worker.tpl").read_text()
        (task / "environment/files/adapters/worker.py").write_text(worker_tpl.replace("TARGET", target))
    else:
        filename = ADAPTERS[language]
        (task / "environment/files/adapters" / filename).write_text(render_worker(problem, language))

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
        for row in problems:
            for language in languages:
                if row["question_id"] in OMITTED_QUESTIONS:
                    exclusions.append({"question_id": row["question_id"], "language": language,
                                       "reason": OMITTED_QUESTIONS[row["question_id"]]})
                    continue
                try:
                    generate(row, language, staged, images.get(language), with_oracle)
                except NotImplementedError as exc:
                    if not skip_unsupported:
                        raise
                    exclusions.append({"question_id": row["question_id"], "language": language, "reason": str(exc)})
        if exclusions:
            (staged / "exclusions.json").write_text(json.dumps(exclusions, indent=2) + "\n")
        staged.rename(output)
    return exclusions, sum(1 for p in output.iterdir() if p.is_dir())