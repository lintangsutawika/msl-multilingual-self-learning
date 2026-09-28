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


def load_problems_hf(
    split: str = "test",
    dataset: str | None = None,
) -> list[dict[str, Any]]:
    """Return the flat per-(problem, language) rows for a LeetCode split.

    Pulls train/test from the hosted neulab/leetcode dataset (built by
    hf_dataset/build_dataset.py). An explicit `dataset` overrides the repo name.
    """
    name = dataset or NEULAB_HF_DATASET
    try:
        ds = load_dataset(name, split=split)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {name!r} split {split!r}. Push it first via "
            f"hf_dataset/build_dataset.py. Original: {exc}"
        ) from exc
    return [dict(row) for row in ds]
