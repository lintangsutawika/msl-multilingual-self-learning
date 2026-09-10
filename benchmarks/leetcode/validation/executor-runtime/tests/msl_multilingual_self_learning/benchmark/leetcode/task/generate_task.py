from __future__ import annotations

import shutil
from pathlib import Path

from ..interfaces.resolve import resolve_interface
from .generate_test import render_test_py
from .prompt import render_prompt


SUPPORTED_LANGUAGES = {
    "python",
    "cpp",
    "go",
    "java",
}


RUNTIME_FILES = [
    "__init__.py",
    "benchmark/__init__.py",
    "benchmark/leetcode/__init__.py",

    "benchmark/leetcode/interfaces/__init__.py",
    "benchmark/leetcode/interfaces/models.py",

    "benchmark/leetcode/executors/__init__.py",
    "benchmark/leetcode/executors/base.py",
    "benchmark/leetcode/executors/native.py",
    "benchmark/leetcode/executors/python.py",
    "benchmark/leetcode/executors/cpp.py",
    "benchmark/leetcode/executors/go.py",
    "benchmark/leetcode/executors/java.py",

    "benchmark/leetcode/verifier/__init__.py",
    "benchmark/leetcode/verifier/verifier.py",
]


def _copy_verifier_runtime(
    tests_dir: Path,
) -> None:
    """
    Copy the minimal verifier package into the generated Harbor task.

    Harbor executes the verifier inside an isolated environment, so the
    repository package is copied under /tests and exposed through PYTHONPATH.
    """

    package_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    destination_root = (
        tests_dir
        / "msl_multilingual_self_learning"
    )

    for relative_path in RUNTIME_FILES:
        source = (
            package_root
            / relative_path
        )

        if not source.exists():
            raise FileNotFoundError(
                "Required verifier runtime file "
                f"does not exist: {source}"
            )

        destination = (
            destination_root
            / relative_path
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source,
            destination,
        )


def _render_test_sh(
    language: str,
) -> str:
    """
    Render the Harbor verifier entry point.

    The Singularity HPC backend only uses the Dockerfile FROM image and does
    not execute Dockerfile RUN layers. Therefore native-language dependencies
    are installed here, inside the writable task environment, immediately
    before the verifier runs.

    test.py exit codes:
        0 -> PASS
        1 -> WRONG_ANSWER
        2+ -> RUNTIME_ERROR
    """

    if language == "python":
        setup = """
true
"""

    elif language == "cpp":
        setup = """
apt-get update
apt-get install -y --no-install-recommends \\
    g++ \\
    nlohmann-json3-dev
"""

    elif language == "go":
        setup = """
apt-get update
apt-get install -y --no-install-recommends \\
    golang-go
"""

    elif language == "java":
        setup = """
apt-get update
apt-get install -y --no-install-recommends \\
    openjdk-17-jdk-headless \\
    libgoogle-gson-java
"""

    else:
        raise ValueError(
            f"Unsupported language: {language}"
        )

    return f"""#!/usr/bin/env bash
set -uo pipefail

mkdir -p /logs/verifier

export PYTHONPATH="/tests:${{PYTHONPATH:-}}"

{setup}

SETUP_STATUS=$?

if [ "$SETUP_STATUS" -ne 0 ]; then
    echo "0" > /logs/verifier/reward.txt
    echo "RUNTIME_ERROR" > /logs/verifier/status.txt

    echo "Failed to install verifier dependencies" \
        > /logs/verifier/test-stdout.txt

    exit 0
fi

python3 /tests/test.py \
    > /logs/verifier/test-stdout.txt \
    2>&1

STATUS=$?

case "$STATUS" in
    0)
        echo "1" > /logs/verifier/reward.txt
        echo "PASS" > /logs/verifier/status.txt
        ;;

    1)
        echo "0" > /logs/verifier/reward.txt
        echo "WRONG_ANSWER" > /logs/verifier/status.txt
        ;;

    *)
        echo "0" > /logs/verifier/reward.txt
        echo "RUNTIME_ERROR" > /logs/verifier/status.txt
        ;;
esac

exit 0
"""


def _render_task_toml(
    problem,
    language: str,
) -> str:
    difficulty = str(
        problem.difficulty
    ).lower()

    return f"""schema_version = "1.0"

[task]
name = "leetcode/{problem.task_id}-{language}"

[metadata]
language = "{language}"
difficulty = "{difficulty}"
source = "newfacade/LeetCodeDataset"
task_id = "{problem.task_id}"

[agent]
timeout_sec = 300

[verifier]
timeout_sec = 60

[environment]
cpus = 1
"""


def _render_dockerfile(
    language: str,
) -> str:
    """
    The Singularity HPC backend reads the Dockerfile's FROM image but does not
    build its RUN layers. We therefore use Python as the common base image
    because the verifier itself is written in Python.

    Language-specific compiler/runtime dependencies are installed by test.sh.
    """

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {language}"
        )

    return """FROM python:3.12-bookworm

WORKDIR /workspace

CMD ["bash"]
"""


def generate_task(
    problem,
    language: str,
    output_dir: Path,
) -> Path:
    """
    Generate one Harbor-compatible LeetCode task.

    Generated structure:

        <task>/
        ├── instruction.md
        ├── task.toml
        ├── environment/
        │   └── Dockerfile
        └── tests/
            ├── test.py
            ├── test.sh
            └── msl_multilingual_self_learning/
                └── ...

    Existing generated task directories are removed before regeneration so
    stale verifier files, Dockerfiles, solutions, or bytecode cannot survive.
    """

    language = language.lower()

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            "Unsupported language: "
            f"{language!r}. "
            "Expected one of: "
            + ", ".join(
                sorted(
                    SUPPORTED_LANGUAGES
                )
            )
        )

    interface = resolve_interface(
        problem,
        language,
    )

    task_dir = (
        Path(output_dir)
        / language
        / problem.task_id
    )

    if task_dir.exists():
        shutil.rmtree(
            task_dir
        )

    environment_dir = (
        task_dir
        / "environment"
    )

    tests_dir = (
        task_dir
        / "tests"
    )

    environment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tests_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    instruction = render_prompt(
        problem,
        interface,
    )

    test_py = render_test_py(
        problem,
        interface,
    )

    test_sh = _render_test_sh(
        language
    )

    task_toml = _render_task_toml(
        problem,
        language,
    )

    dockerfile = _render_dockerfile(
        language
    )

    (
        task_dir
        / "instruction.md"
    ).write_text(
        instruction.rstrip()
        + "\n",
        encoding="utf-8",
    )

    (
        task_dir
        / "task.toml"
    ).write_text(
        task_toml,
        encoding="utf-8",
    )

    (
        environment_dir
        / "Dockerfile"
    ).write_text(
        dockerfile,
        encoding="utf-8",
    )

    (
        tests_dir
        / "test.py"
    ).write_text(
        test_py,
        encoding="utf-8",
    )

    test_sh_path = (
        tests_dir
        / "test.sh"
    )

    test_sh_path.write_text(
        test_sh,
        encoding="utf-8",
    )

    test_sh_path.chmod(
        0o755
    )

    _copy_verifier_runtime(
        tests_dir
    )

    return task_dir