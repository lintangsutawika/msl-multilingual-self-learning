from __future__ import annotations

from pathlib import Path


DOOCS_ROOT = Path("external/doocs-leetcode")

LANGUAGE_EXTENSIONS = {
    "python": ".py",
    "cpp": ".cpp",
    "javascript": ".js",
    "typescript": ".ts",
    "c": ".c",
    "go": ".go",
    "java": ".java",
    "php": ".php",
    "ruby": ".rb",
    "rust": ".rs",
}


def get_problem_range(question_id: int) -> str:
    """
    Return the Doocs range directory containing a LeetCode problem.

    Examples:
        1    -> 0000-0099
        99   -> 0000-0099
        100  -> 0100-0199
        3195 -> 3100-3199
    """
    start = (question_id // 100) * 100
    end = start + 99

    return f"{start:04d}-{end:04d}"


def find_problem_dir(
    question_id: int,
    doocs_root: Path = DOOCS_ROOT,
) -> Path | None:
    """
    Find the Doocs directory corresponding to a LeetCode question ID.

    Matching is based on the numeric LeetCode problem ID, not the title
    or slug.

    Example:
        question_id=1
        -> external/doocs-leetcode/solution/0000-0099/0001.Two Sum
    """
    range_dir = (
        doocs_root
        / "solution"
        / get_problem_range(question_id)
    )

    if not range_dir.exists():
        return None

    prefix = f"{question_id:04d}."

    matches = sorted(
        path
        for path in range_dir.iterdir()
        if path.is_dir()
        and path.name.startswith(prefix)
    )

    if not matches:
        return None

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple Doocs directories found for question "
            f"{question_id}: {matches}"
        )

    return matches[0]


def find_readme(
    question_id: int,
    doocs_root: Path = DOOCS_ROOT,
) -> Path | None:
    """
    Return the README_EN.md path for a Doocs problem.

    This README is the preferred source for language-specific
    callable interfaces because it may contain language sections
    even when standalone Solution.<ext> files are missing.
    """
    problem_dir = find_problem_dir(
        question_id,
        doocs_root=doocs_root,
    )

    if problem_dir is None:
        return None

    readme = problem_dir / "README_EN.md"

    if not readme.exists():
        return None

    return readme


def read_readme(
    question_id: int,
    doocs_root: Path = DOOCS_ROOT,
) -> str | None:
    """
    Read README_EN.md for a problem.

    Returns:
        README text if present, otherwise None.
    """
    readme = find_readme(
        question_id,
        doocs_root=doocs_root,
    )

    if readme is None:
        return None

    return readme.read_text(
        encoding="utf-8",
        errors="replace",
    )


def find_solution_file(
    question_id: int,
    language: str,
    doocs_root: Path = DOOCS_ROOT,
) -> Path | None:
    """
    Find a standalone Doocs Solution.<ext> file.

    This is kept mainly for auditing/debugging.

    It should NOT be treated as the primary interface source because
    README_EN.md often contains language implementations that do not
    have standalone Solution.<ext> files.
    """
    if language not in LANGUAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported languages: "
            f"{', '.join(sorted(LANGUAGE_EXTENSIONS))}"
        )

    problem_dir = find_problem_dir(
        question_id,
        doocs_root=doocs_root,
    )

    if problem_dir is None:
        return None

    extension = LANGUAGE_EXTENSIONS[language]

    solution_file = (
        problem_dir
        / f"Solution{extension}"
    )

    if not solution_file.exists():
        return None

    return solution_file


def get_doocs_commit(
    doocs_root: Path = DOOCS_ROOT,
) -> str | None:
    """
    Return the Git commit SHA of the local Doocs checkout.

    This is useful for reproducibility so interface metadata can be
    tied to the exact upstream revision used to generate it.

    Returns None if the checkout is not a Git repository or the
    commit cannot be determined.
    """
    git_dir = doocs_root / ".git"

    if not git_dir.exists():
        return None

    head = git_dir / "HEAD"

    if not head.exists():
        return None

    head_text = head.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()

    # Detached HEAD:
    # .git/HEAD directly contains the SHA.
    if not head_text.startswith("ref: "):
        return head_text or None

    ref_name = head_text.removeprefix(
        "ref: "
    ).strip()

    ref_path = git_dir / ref_name

    if ref_path.exists():
        sha = ref_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()

        return sha or None

    # In some Git repositories refs are packed.
    packed_refs = git_dir / "packed-refs"

    if packed_refs.exists():
        for line in packed_refs.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if line.startswith("^"):
                continue

            parts = line.split()

            if len(parts) != 2:
                continue

            sha, packed_ref_name = parts

            if packed_ref_name == ref_name:
                return sha

    return None