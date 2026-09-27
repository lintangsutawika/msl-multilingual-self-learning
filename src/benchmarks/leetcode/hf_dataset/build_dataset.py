#!/usr/bin/env python3
"""Build the neulab/leetcode Hugging Face dataset from LeetCode + GraphQL.

Produces the post-interface rows, **flattened to one row per (problem, language)**,
written as JSONL into the target HF repo. To stay under HF's 10 MiB/file limit without
LFS, each split is emitted as one JSONL per language under `data/<split>/<lang>.jsonl`.
HF auto-globs `data/<split>/*.jsonl` into a SINGLE split, so
`load_dataset("neulab/leetcode", split="test")` returns all 1809 rows concatenated,
with the `language` column distinguishing the rows.

Usage:
    python src/benchmarks/leetcode/hf_dataset/build_dataset.py \
        [--out /home/aci18914wh/leetcode] [--only test]

After writing data/{train,test}/<lang>.jsonl + README.md, `git add . && git push`
publishes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root

from src.benchmarks.leetcode.adapter import LANGUAGES  # noqa: E402
from src.benchmarks.leetcode.dataset import load_problems_hf  # noqa: E402

DEFAULT_OUT = Path("/home/aci18914wh/leetcode")
SPLITS = ("train", "test")


def flatten(records: list[dict]) -> list[dict]:
    """Expand nested problem-records into one row per (problem, language)."""
    flat: list[dict] = []
    for rec in records:
        interfaces = rec.get("interfaces")
        if not interfaces:
            flat.append(rec)              # already a flat row; keep as-is
            continue
        base = {k: v for k, v in rec.items() if k != "interfaces"}
        for language, interface in interfaces.items():
            row = dict(base)
            row["language"] = language
            row["interface"] = interface
            flat.append(row)
    return flat


def build(split: str, out: Path) -> None:
    records = load_problems_hf(split)
    if not records:
        print(f"[build] no records for {split}", file=sys.stderr)
        return
    flat = flatten(records)
    split_dir = out / "data" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    # Group by language -> one small JSONL per language (each well under 10 MiB).
    by_lang: dict[str, list[dict]] = {}
    for row in flat:
        by_lang.setdefault(row.get("language"), []).append(row)
    total = 0
    for lang, rows in sorted(by_lang.items()):
        path = split_dir / f"{lang}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        total += len(rows)
        print(f"[build] {split}/{lang}: {len(rows)} rows -> {path}")
    print(f"[build] {split}: {len(flat)} problems -> {total} rows total")


def write_readme(out: Path) -> None:
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
and `metadata`. Stored as `data/<split>/<lang>.jsonl` (one JSONL per language, so
each file stays well under HF's 10 MiB limit); HF globs them all into a single split.

## Splits

- `train`: 2878 problems x 9 languages.
- `test`: 201 problems x 9 languages = 1809 rows (held-out eval).

## Load

```python
from datasets import load_dataset
train = load_dataset("neulab/leetcode", split="train")  # one Dataset, all rows
test  = load_dataset("neulab/leetcode", split="test")   # one Dataset, 1809 rows
python_rows = test.filter(lambda r: r["language"] == "python")
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
    print("[build] done. Push to HF: cd <out> && git add . && git commit && git push")


if __name__ == "__main__":
    main()