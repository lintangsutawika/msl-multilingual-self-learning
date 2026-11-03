"""Shared layout and helpers for the MLE training-data pipeline.

A pipeline *root* holds one directory per task in MLE-bench layout (see README.md):

    <root>/<task_id>/
        config.yaml                      id, source, metric direction, id column, split + sanity results
        grader/grade.py                  grade(submission, answers) -> float, HIGHER_IS_BETTER (+ its own deps)
        leaderboard.csv                  optional: Kaggle private leaderboard (teamName, score)
        prepared/public/                 what the agent sees: description.md, train/test inputs, sample submission
        prepared/private/answers.csv     answers for every test row
        prepared/private/public_answers.csv, private_answers.csv   written by `split`
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
GRADER_SRC = HERE / "grader"
TEMPLATE = HERE / "task-template"

PUBLIC_FRAC = 0.3
SEED = 0

# Files never copied into prepared/public (answers or graders shipped next to the data by some sources).
NEVER_PUBLIC = {"answer.csv", "answers.csv", "submission.csv", "evaluator.py", "generate_task_env.py", "threshold.json"}


def link_file(src: Path, dst: Path) -> None:
    """Hardlink src to dst (same filesystem, no extra space); fall back to a copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def link_tree(src: Path, dst: Path, skip: set[str] = frozenset()) -> None:
    """Mirror a directory tree with hardlinks, skipping files whose name is in `skip`."""
    for p in sorted(src.rglob("*")):
        if p.is_file() and not p.is_symlink() and p.name not in skip:
            link_file(p, dst / p.relative_to(src))


def read_config(task: Path) -> dict:
    return yaml.safe_load((task / "config.yaml").read_text()) or {}


def write_config(task: Path, cfg: dict) -> None:
    (task / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


def find_sample(public: Path) -> Path | None:
    """The sample submission file in a public dir (names vary: sample_submission.csv, sampleSubmission.csv, ...)."""
    cands = sorted(p for p in public.glob("*.csv") if "sample" in p.name.lower() and "sub" in p.name.lower())
    return cands[0] if cands else None


EXCLUDE_FILE = HERE / "mlebench_exclude.txt"


def mlebench_ids() -> set[str]:
    """Lower-cased MLE-bench competition ids; training tasks must never be one of these (eval contamination)."""
    lines = EXCLUDE_FILE.read_text().splitlines()
    return {l.strip().lower() for l in lines if l.strip() and not l.startswith("#")}


def is_mlebench(task_id: str) -> bool:
    return task_id.lower() in mlebench_ids()


def task_dirs(root: Path, only: list[str] | None = None) -> list[Path]:
    """Task dirs under root, never including an MLE-bench competition (even if one was imported by mistake)."""
    out = sorted(p.parent for p in root.glob("*/config.yaml") if not is_mlebench(p.parent.name))
    return [p for p in out if not only or p.name in only]


def dir_mb(d: Path) -> float:
    return round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6, 1)
