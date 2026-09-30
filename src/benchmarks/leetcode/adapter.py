"""Adapter: generate Harbor leetcode tasks from per-language templates.

For each (problem, language) the per-language template directory
(task-template-<lang>) is copied, its {{ }} placeholders are substituted with
problem-specific values, and the problem-specific verifier artifacts (judge
runtime, canonical tests, parameter config, native runner/worker) are rendered
into the task. Mirrors material-harbor/matqna's adapter pattern.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .hf_dataset.constraints import MIN_VALID_TESTS, canonical_names, filter_canonical_tests
from .runners import render_worker

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
    "python": "worker.py",
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


# Constraint audits of the canonical tests, one per split (see constraints.py).
INVALID_TESTS_DIR = Path("benchmarks/leetcode/data")


def invalid_tests_path(split: str) -> Path:
    return INVALID_TESTS_DIR / f"invalid_tests_{split}.json"


def load_invalid_tests(path: Path) -> dict[int, dict[str, list[str]]]:
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
    row: dict[str, Any],
    output: Path,
    docker_image: str | None = None,
) -> Path:
    """Render one task from a flat (problem, language) row into a fresh subdir."""
    language = row["language"]
    interface = row["interface"]
    all_types = [p["type"] for p in interface["parameters"]] + [interface["return_type"]]
    if any(any(t in value for t in ("TreeNode", "ListNode", "Node")) for value in all_types):
        raise NotImplementedError("Object transport is not supported yet")
    if interface["return_type"] in ("void", "None", "NoneType", ""):
        raise NotImplementedError("In-place/void outputs require mutation transport")

    names = canonical_names(row)
    if len(names) != len(interface["parameters"]):
        raise ValueError("Canonical/native parameter counts differ")

    task = output / f"{row['question_id']}-{language}"
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
        "question_id": row["question_id"],
        "language": language,
        "docker_image": str(Path(docker_image).resolve()) if docker_image else "",
        "entrypoint": interface["raw_signature"].strip(),
        "container": container,
        "problem": row["problem_description"].strip(),
        "source_file": source_file,
    }
    _fill(task / "task.toml", **fills)
    _fill(task / "instruction.md", **fills)

    # Problem-specific verifier artifacts.
    (task / "tests/config.json").write_text(json.dumps({
        "language": language,
        "parameter_names": names,
        "callable": interface["callable"],
        "container": interface.get("container"),
    }))
    (task / "tests/canonical_test.py").write_text(row["canonical_tests"]["source"])

    # Native runner/worker (unified: python -> worker.py, others -> runner.<lang>).
    adapter_dir = task / "environment" / "files" / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    filename = ADAPTERS[language]
    (adapter_dir / filename).write_text(render_worker(row, language))

    return task


def _problem_view(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One problem across all its language rows: what the test filter needs."""
    first = rows[0]
    return {
        "question_id": first["question_id"],
        "metadata": first.get("metadata") or {},
        "canonical_tests": first["canonical_tests"],
        "interfaces": {row["language"]: row["interface"] for row in rows},
    }


def generate_all(
    rows: list[dict[str, Any]],
    output: Path,
    images: dict[str, str] | None = None,
    skip_unsupported: bool = False,
    invalid_tests: dict[int, dict[str, list[str]]] | None = None,
    languages: tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Generate one task per flat (problem, language) row, staging atomically.

    Test cases are filtered per problem using all of its language rows, so the
    kept tests do not depend on which languages are generated (`languages`
    restricts the rows turned into tasks). Returns (exclusions, count). Aborts
    (raising) on unsupported transports unless skip_unsupported, in which case
    they are recorded as exclusions.
    """
    images = images or {}
    output.parent.mkdir(parents=True, exist_ok=True)
    by_problem: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_problem.setdefault(int(row["question_id"]), []).append(row)
    exclusions: list[dict[str, Any]] = []
    dropped_tests: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / "tasks"
        staged.mkdir()
        for qid in sorted(by_problem):
            problem_rows = by_problem[qid]
            wanted = [r for r in problem_rows if languages is None or r["language"] in languages]
            if not wanted:
                continue
            if qid in OMITTED_QUESTIONS:
                exclusions.extend({"question_id": qid, "language": r["language"], "reason": OMITTED_QUESTIONS[qid]}
                                  for r in wanted)
                continue
            try:
                source, dropped, kept = filter_canonical_tests(
                    _problem_view(problem_rows), (invalid_tests or {}).get(qid))
            except (KeyError, ValueError) as exc:
                if not skip_unsupported:
                    raise
                exclusions.extend({"question_id": qid, "language": r["language"], "reason": f"Unreadable tests: {exc}"}
                                  for r in wanted)
                continue
            if dropped:
                dropped_tests.append({"question_id": qid, "kept": kept, "dropped": dropped})
            if kept < MIN_VALID_TESTS:
                exclusions.extend({"question_id": qid, "language": r["language"],
                                   "reason": f"Only {kept} valid test cases (< {MIN_VALID_TESTS})"} for r in wanted)
                continue
            for row in wanted:
                if dropped:
                    row = {**row, "canonical_tests": {**row["canonical_tests"], "source": source}}
                try:
                    generate(row, staged, images.get(row["language"]))
                # NotImplementedError: unsupported transport; ValueError: the dataset's
                # canonical tests and native interface disagree (e.g. parameter counts).
                except (NotImplementedError, ValueError) as exc:
                    if not skip_unsupported:
                        raise
                    exclusions.append({"question_id": qid, "language": row["language"], "reason": str(exc)})
        if exclusions:
            (staged / "exclusions.json").write_text(json.dumps(exclusions, indent=2) + "\n")
        if dropped_tests:
            (staged / "dropped_tests.json").write_text(json.dumps(dropped_tests, indent=2) + "\n")
        staged.rename(output)
    return exclusions, sum(1 for p in output.iterdir() if p.is_dir())
