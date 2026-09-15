"""Per-language instruction prompt builder (programmatic form of the templates).

Kept as a function so tests and tooling can build an instruction without the
file-copy path. Mirrors the `instruction.md` templates in task-template-<lang>.
"""
from __future__ import annotations

from typing import Any

SOURCE_FILES = {
    "python": "solution.py",
    "cpp": "solution.cpp",
    "go": "solution.go",
    "java": "Solution.java",
}


def format_entrypoint_requirement(language: str, interface: dict[str, Any]) -> str:
    return (
        f"Entrypoint: {interface['raw_signature'].strip()}\n"
        f"Container: {interface.get('container') or '(none)'}"
    )


def build_prompt(record: dict[str, Any], language: str) -> str:
    """One instruction for all languages; no starter code or tests.

    The test suite asserts that everything after `Problem:\\n` is identical across
    languages (the shared problem statement), so language-specific guidance is
    kept before the Problem block and the post-problem text is language-neutral.
    """
    interface = record["interfaces"][language]
    source_file = SOURCE_FILES[language]
    return (
        f"Language: {language}\n"
        f"{format_entrypoint_requirement(language, interface)}\n\n"
        "Solve this LeetCode problem in the specified language. Write a complete "
        "source file, including any required imports, includes, or package "
        "declarations. Do not define a program entrypoint. The source must compile "
        "or run as submitted. Do not include unused imports.\n\n"
        f"Save your final solution to /workspace/{source_file} (the working "
        "directory is /workspace). Write only that source file, without Markdown "
        "fences or explanations.\n\n"
        "When the file is written and you are done, issue the command:\n"
        "    echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n\n"
        f"Problem:\n{record['problem_description'].strip()}"
    )