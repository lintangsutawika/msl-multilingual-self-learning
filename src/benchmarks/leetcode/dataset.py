"""Load the hosted neulab/leetcode dataset (pure puller; no build machinery).

The actual dataset construction (LeetCode + GraphQL -> post-interface rows) lives in
hf_dataset/build_dataset.py; the output is pushed to HF (neulab/leetcode) and this
module just pulls it. ``load_problems_hf(split)`` returns the flat per-(problem,
language) rows (one row per language), which the task generator (adapter.py) consumes.
"""
from __future__ import annotations

from typing import Any

from datasets import load_dataset

NEULAB_HF_DATASET = "neulab/leetcode"


def _flatten_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand nested problem-records (interfaces={lang: interface}) into one row
    per (problem, language): each row has a single `language` + `interface`."""
    flat: list[dict[str, Any]] = []
    for rec in records:
        interfaces = rec.get("interfaces")
        if not interfaces:
            flat.append(rec)
            continue
        base = {k: v for k, v in rec.items() if k != "interfaces"}
        for language, interface in interfaces.items():
            row = dict(base)
            row["language"] = language
            row["interface"] = interface
            flat.append(row)
    return flat


def load_problems_from_hf_repo(split: str = "test") -> list[dict[str, Any]] | None:
    """Return problem rows from the hosted neulab/leetcode dataset.

    Fast path: no GraphQL, no build -- rows were pushed by hf_dataset/build_dataset.py.
    Returns None if the hosted dataset isn't available.
    """
    try:
        ds = load_dataset(NEULAB_HF_DATASET, split=split)
    except Exception:
        return None
    return [dict(row) for row in ds]


def load_problems_hf(split: str = "test") -> list[dict[str, Any]]:
    """Return problem rows for a LeetCode split (test | train) from the hosted
    neulab/leetcode dataset. Raises if the split isn't published yet."""
    hosted = load_problems_from_hf_repo(split)
    if hosted is None:
        raise RuntimeError(
            f"Hosted dataset {NEULAB_HF_DATASET!r} not available for split "
            f"{split!r}. Push it first via hf_dataset/build_dataset.py."
        )
    if hosted and "interfaces" in hosted[0]:
        return _flatten_rows(hosted)
    return hosted
