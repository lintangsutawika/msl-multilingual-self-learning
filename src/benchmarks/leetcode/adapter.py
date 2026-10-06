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

from .hf_dataset.constraints import canonical_names
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


def _source_contract(interface: dict, language: str) -> str:
    """A consistent, per-language statement of the exact entrypoint shape the
    grader's runner invokes, derived from the `interface` field.

    Consistent across tasks within each language, so the model always knows the
    required symbol/package/class instead of guessing (`package main`, `candidate`,
    a `main` func, etc.) and failing the compile/entrypoint check.
    """
    cont = (interface.get("container") or "").strip()
    call = (interface.get("callable") or "").strip()
    if language == "go":
        return (
            f"Declare the function `func {call}(...)` inside `package solution` "
            f"(NOT `package main`, and do not define `func main` -- the grader "
            f"runs its own driver)."
        )
    if language == "rust":
        return (
            f"Declare `struct {cont or 'Solution'}` and implement "
            f"`impl {cont or 'Solution'} {{ fn {call}(...) -> ... }}` "
            f"(the runner invokes `{cont or 'Solution'}::{call}(...)`)."
        )
    if cont:
        return (
            f"Declare a class/type `{cont}` exposing a method/function named "
            f"`{call}` (the runner invokes it as `{cont}.{call}(...)` or "
            f"`{cont}().{call}(...)`)."
        )
    if language == "javascript" or language == "typescript":
        return (
            f"Declare the function `{call}` (the runner invokes `{call}(...)`)."
        )
    if language == "ruby":
        return (
            f"Declare the method `def {call}(...)` (the runner invokes `{call}(...)`)."
        )
    # fallback: python / anything with no container
    return (
        f"Declare the entrypoint `{call}` exactly as given (match the signature "
        f"and parameter names)."
    )


def _extract_public_cases(canonical_tests: dict) -> str:
    """Extract the public/example test cases from the canonical Python judge source.

    The neulab/leetcode dataset stores the public cases as ``assert candidate(...)``
    lines in ``canonical_tests.source`` (the same check() oracle used for grading),
    so surfacing them in the agent's instructions gives concrete input->expected
    examples. Returns a single formatted block (empty if none found).
    """
    source = (canonical_tests or {}).get("source") or ""
    lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith("assert candidate(")
    ]
    if not lines:
        return ""
    # Bound the inlined cases so the agent's --task argv stays well under the OS
    # per-argument exec limit (Linux MAX_ARG_STRLEN = 128 KB): a handful of cases
    # with huge inputs (e.g. 10k-element arrays) can exceed it and fail the sandbox
    # exec with "OSError: [Errno 7] Argument list too long". Keep the full set on
    # disk (tests/canonical_test.py) and inline a bounded prefix + pointer.
    max_bytes = 96 * 1024  # safely under the 128 KB single-argument limit
    block = []
    used = 0
    for line in lines:
        if used + len(line) > max_bytes and block:
            break
        block.append(line)
        used += len(line) + 1
    out = "\n".join(block)
    omitted = len(lines) - len(block)
    if omitted:
        out += f"\n# ...and {omitted} more public cases in tests/canonical_test.py"
    return out


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


def prebuild_sif(
    dockerfile: Path,
    out_sif: Path,
    *,
    container_bin: str = "singularity",
    force: bool = False,
) -> Path:
    """Build a Singularity sif from a task Dockerfile (FROM + RUN/COPY/ENV deps).

    Writes a .def alongside out_sif, runs ``singularity build --fakeroot`` with the
    Dockerfile's dir as the build context (so COPY sources resolve), and returns
    out_sif. Raises CalledProcessError on a failed build.

    Idempotent: if ``out_sif`` already exists and ``force`` is False, the build is
    skipped and the existing sif is returned. Pass ``force=True`` to rebuild.
    """
    dockerfile = Path(dockerfile).resolve()
    out_sif = Path(out_sif).resolve()
    if out_sif.is_file() and not force:
        print(f"[prebuild] {out_sif.name} exists; skipping (--prebuild-force to rebuild)")
        return out_sif
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
    force: bool = False,
) -> dict[str, str]:
    """Build one sif per language from its template Dockerfile, into out_dir.

    Returns a dict {language: <sif path>} suitable for adapter.generate_all's
    images arg (which becomes task.toml [environment].docker_image). Skips any
    language whose sif already exists unless ``force`` is True (rebuild)."""
    template_dir = template_dir or PKG
    images: dict[str, str] = {}
    for lang in languages:
        dockerfile = template_dir / f"task-template-{lang}/environment/Dockerfile"
        if not dockerfile.exists():
            raise FileNotFoundError(f"No template Dockerfile for {lang}: {dockerfile}")
        sif = Path(out_dir) / f"{lang}.sif"
        prebuild_sif(dockerfile, sif, container_bin=container_bin, force=force)
        images[lang] = str(sif)
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
        # A canonical/native parameter-count mismatch marks an unsupported
        # object-transport interface (e.g. Java TreeNode/ListNode methods parse
        # to empty params, or Rust tree methods expose a single Self root while
        # the canonical signature has more). Treat it as unsupported so
        # --skip-unsupported records it in exclusions.json instead of aborting.
        raise NotImplementedError(
            "Object transport is not supported yet "
            "(canonical/native parameter counts differ)"
        )

    task = output / f"{row['question_id']}-{language}"
    if task.exists():
        raise FileExistsError(f"Use a fresh output directory: {task}")

    # Copy the per-language template into the new task.
    tpl = PKG / f"task-template-{language}"
    shutil.copytree(tpl, task, dirs_exist_ok=False, ignore=shutil.ignore_patterns("__pycache__"))

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
        "public_cases": _extract_public_cases(row.get("canonical_tests")),
        "source_contract": _source_contract(interface, language),
        "raw_signature": interface["raw_signature"].strip(),
        "callable": interface.get("callable", ""),
        # lang used by the stub fence in instruction.md
        "lang": language,
        # The pre-declared stub source, shown in the instruction so the agent sees
        # the exact entrypoint despite /workspace starting empty. Pre-fill the
        # nested {raw_signature}/{callable} placeholders: _fill does a single pass
        # and the stub is inserted after raw_signature is already substituted.
        "stub": (
            (task / "solution" / source_file).read_text()
            .replace("{raw_signature}", interface["raw_signature"].strip())
            .replace("{callable}", interface.get("callable", ""))
            if (task / "solution" / source_file).exists() else ""
        ),
    }
    _fill(task / "task.toml", **fills)
    _fill(task / "instruction.md", **fills)
    # Pre-fill the solution stub: the entrypoint is already declared there, so the
    # agent just edits the body instead of guessing the signature/package/class.
    _fill(task / "solution" / source_file, **fills)
    # Also stage the stub under environment/files/adapters/ (next to the runner):
    # that dir is what actually reaches the container's /workspace (the runner
    # worker.py lives at /workspace/worker.py), so the pre-declared entrypoint
    # lands at /workspace/<source_file> for the agent to edit, on both Singularity
    # and Modal (they both inject the task's environment/files/adapters).
    _adapters = task / "environment" / "files" / "adapters"
    _adapters.mkdir(parents=True, exist_ok=True)
    (_adapters / source_file).write_text(
        (task / "solution" / source_file).read_text()
    )
    # The interface/entrypoint contract as JSON (tests/interface.json), consumed
    # by the verifier: {language, code(of the declared stub), callable, container}.
    _stub_src = (task / "solution" / source_file).read_text()
    (task / "tests" / "interface.json").write_text(
        json.dumps({
            "language": language,
            "code": _stub_src,
            "callable": interface.get("callable", ""),
            "container": container,
        }, ensure_ascii=False)
    )

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


def generate_all(
    rows: list[dict[str, Any]],
    output: Path,
    images: dict[str, str] | None = None,
    skip_unsupported: bool = False,
    languages: tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Generate one task per flat (problem, language) row, staging atomically.

    The rows come from neulab/leetcode, whose tests were already cleaned when
    the dataset was built (hf_dataset/build_dataset.py), so they are used as
    they are. `languages` restricts the rows turned into tasks. Returns
    (exclusions, count). Aborts (raising) on unsupported transports unless
    skip_unsupported, in which case they are recorded as exclusions.
    """
    images = images or {}
    output.parent.mkdir(parents=True, exist_ok=True)
    exclusions: list[dict[str, Any]] = []
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:  # tqdm not yet installed: fall back to a plain counter
        def _tqdm(it, total=None, desc=None, unit="", **unused):  # type: ignore
            return it

    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / "tasks"
        staged.mkdir()
        for row in _tqdm(rows, total=len(rows), desc="generating tasks", unit="task"):
            lang = row["language"]
            if languages is not None and lang not in languages:
                continue
            try:
                generate(row, staged, images.get(lang))
            # NotImplementedError: unsupported transport; ValueError: the dataset's
            # canonical tests and native interface disagree (e.g. parameter counts).
            except (NotImplementedError, ValueError) as exc:
                if not skip_unsupported:
                    raise
                exclusions.append({"question_id": row["question_id"], "language": lang, "reason": str(exc)})
        if exclusions:
            (staged / "exclusions.json").write_text(json.dumps(exclusions, indent=2) + "\n")
        staged.rename(output)
    return exclusions, sum(1 for p in output.iterdir() if p.is_dir())
