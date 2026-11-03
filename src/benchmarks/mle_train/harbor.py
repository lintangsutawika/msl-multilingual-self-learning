"""Turn usable tasks from a pipeline root into Harbor tasks with a public/private leaderboard.

    <out>/<task_id>/
        instruction.md            full task description + metric direction + public/private workflow
        task.toml                 metadata: source, direction, split sizes, sample-submission scores (episode baseline)
        environment/Dockerfile    python:3.12-slim + common ML libs; public data at /workspace/public
        environment/validate      scores a file on the PUBLIC split, logs to /logs/verifier/validation_scores.jsonl
        environment/grader/       grade_cli.py + the task's grader + public_answers.csv  (inside the agent's box)
        tests/test.sh             grades /workspace/predictions.csv on the PRIVATE split, writes reward.txt
        tests/grader/             grade_cli.py + the task's grader + private_answers.csv (+ leaderboard.csv)
        solution/solve.sh         oracle: submits the sample submission

Harbor uploads tests/ only after the agent finishes, so the private answers are never in the agent's box.
The public answers are (validate runs in the box); see README "Known gaps".
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from common import GRADER_SRC, TEMPLATE, is_mlebench, link_file, link_tree, read_config


def _fill(path: Path, kw: dict) -> None:
    """Replace {key} for known keys in one pass (unknown braces, e.g. in f-strings, are left alone)."""
    path.write_text(re.sub(r"\{(\w+)\}", lambda m: str(kw[m.group(1)]) if m.group(1) in kw else m.group(0),
                           path.read_text()))


def build_one(task: Path, out: Path, agent_timeout: int, cpus: int, memory_mb: int) -> Path:
    if is_mlebench(task.name):
        raise ValueError(f"{task.name} is an MLE-bench competition; refusing to build it as a training task")
    cfg = read_config(task)
    t = out / task.name
    if t.exists():
        shutil.rmtree(t)
    shutil.copytree(TEMPLATE, t)
    env, tests = t / "environment", t / "tests"
    link_tree(task / "prepared/public", env / "public")
    for g in (env / "grader", tests / "grader"):
        shutil.copytree(task / "grader", g, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(GRADER_SRC / "grade_cli.py", g / "grade_cli.py")
    link_file(task / "prepared/private/public_answers.csv", env / "grader/public_answers.csv")
    link_file(task / "prepared/private/private_answers.csv", tests / "grader/private_answers.csv")
    s = cfg["sanity"]
    if cfg.get("leaderboard"):
        link_file(task / cfg["leaderboard"], tests / "grader/leaderboard.csv")
        reward_args = "--leaderboard /tests/grader/leaderboard.csv"
    else:  # normalized between the sample submission (0) and the true answers (1) on the private split
        reward_args = f"--baseline {s['private_sample']!r} --ceiling {s['private_perfect']!r}"
    desc_f = task / "prepared/public/description.md"
    hib = bool(cfg["higher_is_better"])
    kw = {
        "task_id": task.name,
        "source": cfg["source"],
        "higher_is_better": str(hib).lower(),
        "id_column": cfg["id_column"],
        "public_ids": cfg["split"]["public_ids"],
        "private_ids": cfg["split"]["private_ids"],
        "public_sample": repr(float(s["public_sample"])),
        "private_sample": repr(float(s["private_sample"])),
        "reward": cfg["reward"],
        "agent_timeout": float(agent_timeout),
        "cpus": cpus,
        "memory_mb": memory_mb,
        "memory_gb": memory_mb // 1024,
        "hours": f"{agent_timeout / 3600:g}",
        "sample_submission": cfg["sample_submission"],
        "direction": "higher is better" if hib else "lower is better",
        "reward_args": reward_args,
        "description": desc_f.read_text(errors="replace").strip() if desc_f.exists() else f"Task {task.name}.",
    }
    for rel in ("task.toml", "instruction.md", "environment/validate", "tests/test.sh", "solution/solve.sh"):
        _fill(t / rel, kw)
    for rel in ("environment/validate", "tests/test.sh", "solution/solve.sh"):
        (t / rel).chmod(0o755)
    return t


def build(root: Path, out: Path, only: list[str] | None, source: str | None, limit: int | None,
          max_public_mb: float, agent_timeout: int, cpus: int, memory_mb: int) -> list[Path]:
    import pandas as pd

    idx = pd.read_csv(root / "index.tsv", sep="\t")
    sel = idx[(idx.usable == True) & ~idx.task.map(is_mlebench)]  # noqa: E712
    if only:
        sel = sel[sel.task.isin(only)]
    if source:
        sel = sel[sel.source == source]
    sel = sel[sel.public_mb <= max_public_mb].sort_values(["public_mb", "task"])
    if limit:
        sel = sel.head(limit)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for name in sel.task:
        made.append(build_one(root / name, out, agent_timeout, cpus, memory_mb))
        print("built", made[-1])
    return made


def export(tasks_dir: Path, out: Path, agent_name: str | None) -> None:
    """Write the training prompt file Miles reads (--prompt-data, see src/run_training.py): one line per Harbor task,
    {"prompt": <instruction.md>, "metadata": {"instance_id": <task dir name>, "source": ...}}, with HARBOR_TASKS_DIR
    pointing at tasks_dir. Also writes <out stem>.<source>.txt task lists (for oracle checks or subsets)."""
    import json
    import tomllib
    from collections import defaultdict

    by_source, n = defaultdict(list), 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for t in sorted(p.parent for p in tasks_dir.glob("*/task.toml")):
            if is_mlebench(t.name):
                raise ValueError(f"{t.name} in {tasks_dir} is an MLE-bench competition; remove it before training")
            meta = tomllib.loads((t / "task.toml").read_text()).get("metadata", {})
            md = {"instance_id": t.name, "source": meta.get("source")}
            if agent_name:
                md["agent_name"] = agent_name
            f.write(json.dumps({"prompt": (t / "instruction.md").read_text(), "metadata": md}) + "\n")
            by_source[meta.get("source")].append(t.name)
            n += 1
    for src, names in by_source.items():
        (out.parent / f"{out.stem}.{src}.txt").write_text("\n".join(names) + "\n")
    print(f"wrote {out} ({n} tasks): " + ", ".join(f"{s}={len(v)}" for s, v in sorted(by_source.items())))
