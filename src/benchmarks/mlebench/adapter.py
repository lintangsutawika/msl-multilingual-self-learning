"""Adapter: generate Harbor MLE-Bench tasks from a single shared task template.

Mirrors src/benchmarks/leetcode/adapter.py: each competition is instantiated by
COPYING the per-benchmark task template (``task-template/``) into a fresh task
dir and substituting ``{placeholder}`` tokens with competition-specific values.

Unlike leetcode (which has one template per language), MLE-Bench has ONE template
shared by all competitions: the environment/Dockerfile installs mlebench+kaggle,
the verifier runs ``mlebench grade``, and per-competition values (id, name,
difficulty, problem, grading metric, Kaggle creds) are filled in at generation.

Generated tasks land under ``<output>/<competition_id>/``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from mlebench.registry import registry

# Template package dir (this file lives in src/benchmarks/mlebench/adapter.py).
PKG = Path(__file__).parent
TEMPLATE = PKG / "task-template"


def _fill(path: Path, **kw: str) -> None:
    """Substitute ``{key}`` placeholders in a file in place (single pass)."""
    if not path.exists():
        return
    text = path.read_text()
    for k, v in kw.items():
        text = text.replace("{" + k + "}", str(v))
    path.write_text(text)


def _metric_notes(comp) -> str:
    grader_name = getattr(getattr(comp, "grader", None), "name", None)
    ct = getattr(comp, "competition_type", "")
    return f"metric={grader_name or 'unknown'} type={ct}"


def _problem_desc(comp) -> str:
    """Return the competition's SANITIZED problem statement for the agent.

    mlebench ships both description.md (full, may leak raw feature/test-set
    details) and description_obfuscated.md (the agent-facing sanitized prompt).
    Always use the obfuscated one; fall back to the full description if a
    competition has no obfuscated file.
    """
    try:
        obf = (
            registry.get_competitions_dir()
            / getattr(comp, "id", comp) / "description_obfuscated.md"
        )
        if obf.exists():
            return obf.read_text().strip()[:2000]
    except Exception:
        pass
    return str(getattr(comp, "description", "") or "").strip()[:2000]


def generate(competition_id: str, difficulty: str, output: Path) -> Path:
    """Instantiate one competition from the shared template into ``output``."""
    comp = registry.get_competition(competition_id)
    task = output / competition_id
    if task.exists():
        # Re-generate idempotently: wipe stale copies so fills are clean.
        shutil.rmtree(task)
    shutil.copytree(TEMPLATE, task, dirs_exist_ok=True)

    fills = {
        "comp_id": competition_id,
        "comp_name": str(getattr(comp, "name", competition_id) or competition_id),
        "difficulty": difficulty,
        "metric_notes": _metric_notes(comp),
        "description": _problem_desc(comp),
        # Bake the Kaggle creds into the image so agent AND verifier Sailboxes
        # can authenticate (`mlebench prepare`). Read from env at generation.
        "kaggle_user": os.environ.get("KAGGLE_USERNAME", ""),
        "kaggle_key": os.environ.get("KAGGLE_KEY", ""),
    }
    for path in (
        task / "task.toml",
        task / "instruction.md",
        task / "environment" / "Dockerfile",
        task / "environment" / "validate_server.py",
        task / "environment" / "validate.py",
        task / "environment" / "startup.sh",
        task / "tests" / "test.sh",
    ):
        _fill(path, **fills)
    for _bin in ("tests/test.sh", "environment/startup.sh", "environment/validate"):
        _p = task / _bin
        if _p.exists():
            _p.chmod(0o755)
    return task


def generate_all(
    targets: dict[str, str],
    output: Path,
) -> list[tuple[str, str, Path]]:
    """Generate all (competition_id -> difficulty) targets under output."""
    output.mkdir(parents=True, exist_ok=True)
    made: list[tuple[str, str, Path]] = []
    for cid, diff in sorted(targets.items()):
        split_out = output / f"mlebench-{diff}"
        split_out.mkdir(parents=True, exist_ok=True)
        d = generate(cid, diff, split_out)
        made.append((cid, diff, d))
    return made