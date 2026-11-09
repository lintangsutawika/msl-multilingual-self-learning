#!/usr/bin/env python3
"""Per-task table of a baseline sweep (slurm/sweep_sail.sbatch): how hard each task is for the model.

    python src/benchmarks/mle_train/sweep_summary.py --job-dir SWEEP_DIR/jobs/NAME --tasks TASKS [--out summary.tsv]

Reads every finished trial (<trial dir>/result.json; the task comes from the trial's config.json) and its verifier files:
- grade.json: valid final submission and medal (MLE-bench rules on the Kaggle leaderboard);
- final_score.json: final score on the held-back half and its leaderboard position (the Harbor reward);
- validation_scores.jsonl: every `validate` score in order. "improvements" counts scores that beat the trial's best
  so far in the metric's direction, after the first: the progression signal the trainer rewards.
Unfinished trials (no result.json) are skipped. Hitting the agent time limit counts as a timeout (normal; the
submission is still graded), any other exception as an error. Tasks where a trial beat the whole leaderboard are
flagged for review: that is what a leak or a non-comparable leaderboard looks like.
"""
import argparse
import json
import statistics
import tomllib
from collections import defaultdict
from pathlib import Path

MEDALS = ("gold", "silver", "bronze")


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def improvements(scores: list[float], higher_is_better: bool) -> int:
    n, best = 0, None
    for s in scores:
        if best is not None and (s > best if higher_is_better else s < best):
            n += 1
        best = s if best is None else (max(best, s) if higher_is_better else min(best, s))
    return n


def trial_row(d: Path, hib: bool) -> dict:
    res = _load(d / "result.json") or {}
    v = d / "verifier"
    grade = _load(v / "grade.json") or {}
    final = _load(v / "final_score.json") or {}
    scores = []
    if (v / "validation_scores.jsonl").exists():
        for line in (v / "validation_scores.jsonl").read_text().splitlines():
            try:
                scores.append(float(json.loads(line)["submission_score"]))
            except Exception:
                pass
    valid = bool(grade.get("valid_submission", final.get("valid")))
    exc = (res.get("exception_info") or {}).get("exception_type")
    return {
        "error": exc is not None and exc != "AgentTimeoutError",  # hitting the time limit is normal, not an error
        "timeout": exc == "AgentTimeoutError",
        "valid": valid,
        "medal": grade.get("medal") if valid else None,
        "position": float(final.get("reward") or 0.0) if valid else 0.0,
        "score": final.get("score") if valid else None,
        "n_validate": len(scores),
        "improvements": improvements(scores, hib),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True, help="Harbor tasks dir (for each task's metric direction)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    hib = {}
    for t in a.tasks.glob("*/task.toml"):
        hib[t.parent.name] = bool(tomllib.loads(t.read_text()).get("metadata", {}).get("higher_is_better", True))
    by_task = defaultdict(list)
    for d in sorted(p for p in a.job_dir.iterdir() if p.is_dir() and "__" in p.name):
        if (d / "result.json").exists():
            # Harbor truncates long task names in trial dir names (covid19-...-week-2/3/4 collide): use the config.
            cfg = _load(d / "config.json") or {}
            task = Path((cfg.get("task") or {}).get("path") or d.name.rsplit("__", 1)[0]).name
            by_task[task].append(trial_row(d, hib.get(task, True)))

    cols = ["task", "trials", "errors", "valid", "medals", "gold", "silver", "bronze", "mean_position",
            "max_position", "mean_validate_calls", "trials_improving", "mean_improvements", "timeouts", "review"]
    lines = ["\t".join(cols)]
    tot = defaultdict(float)
    for task in sorted(by_task):
        rows = by_task[task]
        n = len(rows)
        medal = {m: sum(r["medal"] == m for r in rows) for m in MEDALS}
        vals = {
            "task": task, "trials": n, "errors": sum(r["error"] for r in rows), "valid": sum(r["valid"] for r in rows),
            "medals": sum(medal.values()), **medal,
            "mean_position": round(statistics.mean(r["position"] for r in rows), 4),
            "max_position": round(max(r["position"] for r in rows), 4),
            "mean_validate_calls": round(statistics.mean(r["n_validate"] for r in rows), 1),
            "trials_improving": sum(r["improvements"] > 0 for r in rows),
            "mean_improvements": round(statistics.mean(r["improvements"] for r in rows), 2),
            "timeouts": sum(r["timeout"] for r in rows),
            # Beating every Kaggle team is what a leak or a non-comparable leaderboard looks like: check by hand.
            "review": "beats whole leaderboard" if any(r["position"] >= 0.999 for r in rows) else "",
        }
        lines.append("\t".join(str(vals[c]) for c in cols))
        for k in ("trials", "errors", "valid", "medals", "trials_improving"):
            tot[k] += vals[k]
        tot["position_sum"] += sum(r["position"] for r in rows)
        tot["never_valid"] += vals["valid"] == 0
        tot["any_medal"] += vals["medals"] > 0
    text = "\n".join(lines) + "\n"
    if a.out:
        a.out.write_text(text)
    else:
        print(text)
    n = max(tot["trials"], 1)
    print(f"tasks {len(by_task)} | trials {int(tot['trials'])} | errors {int(tot['errors'])} | "
          f"valid {tot['valid'] / n:.1%} | any medal {tot['medals'] / n:.1%} | mean position {tot['position_sum'] / n:.3f} | "
          f"trials improving on validate {tot['trials_improving'] / n:.1%}")
    print(f"tasks never valid: {int(tot['never_valid'])} | tasks with a medal in some trial: {int(tot['any_medal'])}")
    flagged = [ln.split("\t")[0] for ln in lines[1:] if ln.endswith("beats whole leaderboard")]
    if flagged:
        print("REVIEW (a trial beat the whole Kaggle leaderboard; leak or non-comparable task?):", " ".join(flagged))


if __name__ == "__main__":
    main()
