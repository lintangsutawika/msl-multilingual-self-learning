#!/usr/bin/env python3
"""Build the neulab/leetcode Hugging Face dataset from LeetCode + GraphQL.

Produces the post-interface rows, **flattened to one row per (problem, language)**
(so the test split is 201 problems x 9 languages = 1809 rows). Each row carries a
single `interface` for its `language`, plus the shared `canonical_tests` /
`problem_description` / `metadata`. Writes JSONL into the target HF repo directory
for a git push; `datasets.load_dataset("neulab/leetcode", split=...)` auto-discovers
the `data/{train,test}.jsonl` files, which the task generator (adapter.py) consumes.

Usage:
    python src/benchmarks/leetcode/hf_dataset/build_dataset.py \
        [--out /home/aci18914wh/leetcode] [--only test]

After writing data/{train,test}.jsonl + README.md, `git add . && git push` publishes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root

from src.benchmarks.leetcode.dataset import load_problems_hf  # noqa: E402

DEFAULT_OUT = Path("/home/aci18914wh/leetcode")
SPLITS = ("train", "test")


def flatten(records: list[dict]) -> list[dict]:
    """Expand each nested problem-record into one row per (problem, language).

    A nested record has `interfaces = {lang: interface}`; we emit one row per
    language with `language` + a single `interface`, and carry the shared
    `canonical_tests` / `problem_description` / `metadata` unchanged.
    """
    flat: list[dict] = []
    for rec in records:
        interfaces = rec.pop("interfaces", {})
        for language, interface in interfaces.items():
            row = dict(rec)          # question_id, task_id, difficulty, problem_description, canonical_tests, metadata
            row["language"] = language
            row["interface"] = interface
            flat.append(row)
    return flat


def build(split: str, out: Path) -> None:
    records = load_problems_hf(split)
    if not records:
        print(f"[build] no records for {split}; snippets cache may be unpopulated", file=sys.stderr)
        return
    flat = flatten(records)
    out_dir = out / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in flat:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[build] {split}: {len(records)} problems -> {len(flat)} rows -> {path}")


def write_readme(out: Path) -> None:
    # `language: code` is HF's required special value for programming-language /
    # code datasets (per-language names like python/java/rust are NOT valid ISO codes).
    readme = """---
license: apache-2.0
language:
- code
task_categories:
- text-generation
---

# LeetCode multilingual benchmark dataset

Post-interface rows for the msl-multilingual-self-learning benchmark, flattened to
one row per (problem, language). Each row has the single `interface` for its
`language` (python, cpp, go, java, rust, javascript, typescript, php, ruby), plus
the shared `canonical_tests` (a Python check(candidate) oracle), `problem_description`,
and `metadata`.

## Splits

- `train`: 2878 problems x 9 languages rows (all-9-language universe minus the
  newfacade test split).
- `test`: 201 problems x 9 languages = 1809 rows (held-out eval).

## Load

```python
from datasets import load_dataset
train = load_dataset("neulab/leetcode", split="train")
test = load_dataset("neulab/leetcode", split="test")
```
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(f"[build] wrote {out / 'README.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", choices=SPLITS, default=None)
    args = ap.parse_args()

    splits = (args.only,) if args.only else SPLITS
    for split in splits:
        build(split, args.out)
    write_readme(args.out)
    print("[build] done. Push the repo to HF: cd <out> && git add . && git commit && git push")


if __name__ == "__main__":
    main()