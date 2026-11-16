"""Training monitor for a rollout run: is the policy getting better, and is the loop healthy?

    python3 scripts/monitor_run.py --run-dir <MSL_STORE>/rollout14 --tasks-dir <tasks> [--bin 5]

Reads only finished trials (trials/<trial>/<id>/result.json + verifier/grade.json) and the first
line of each trial's spool file (the weight version the trial started on). Read-only, cheap.

Quality, per bin of --bin weight versions (by the version a trial started on):
  test_q    leaderboard percentile of the final submission on the held-out half (the rows `validate`
            never scores; invalid = 0): the training-side counterpart of the eval's held-out q.
            Raw reward is not this: a better policy improves on its own first try less often.
  first_q   percentile of the trial's first signed validate score (did the first attempt get better?)
  valid     share of trials with a valid final submission (the earliest collapse signal in runs 11-13)
  limit     share stopped by a limit: step cap ("Agent reached maximum") or wall clock (AgentTimeoutError)
  dur_h     median trial hours; vals = median distinct validate scores per trial
Health (last --hours hours): trials finished per hour, launch failures (other exceptions, or ending
within 120 s), and the newest spool write, so an outage like 2026-11-09's SAIL credit stop shows.
"""
import argparse, csv, glob, json, os, re, statistics as st, time
from collections import Counter, defaultdict
from datetime import datetime, timezone

SCORE_RE = re.compile(rb"score=([-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?) sig=[0-9a-f]{16}")
LIMIT_EXC = {"AgentTimeoutError"}
OK_EXC = {None, "AgentTimeoutError", "NonZeroAgentExitCodeError"}  # trials that really ran


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() if s else None


_boards = {}
def board(tasks_dir, task):
    if task not in _boards:
        try:
            with open(os.path.join(tasks_dir, task, "environment", "leaderboard.csv")) as f:
                _boards[task] = sorted(float(r["score"]) for r in csv.DictReader(f)
                                       if r.get("score") not in (None, "") and r.get("hasScore", "True") != "False")
        except (OSError, ValueError, KeyError):
            _boards[task] = None
    return _boards[task]


def q(b, s, lower):
    if s is None or not b:
        return 0.0
    return sum((s < x) if lower else (s > x) for x in b) / len(b)


def start_version(spool, trial):
    try:
        with open(os.path.join(spool, trial + ".jsonl"), "rb") as f:
            return json.loads(f.readline()).get("weight_version")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tasks-dir", required=True)
    ap.add_argument("--bin", type=int, default=5)
    ap.add_argument("--hours", type=float, default=6)
    a = ap.parse_args()
    spool = os.path.join(a.run_dir, "spool")
    rows, health = [], []
    for r in glob.glob(os.path.join(a.run_dir, "trials", "*", "*", "result.json")):
        d = os.path.dirname(r); trial = os.path.basename(os.path.dirname(d)); task = trial.split("__")[0]
        try:
            res = json.load(open(r))
        except Exception:
            continue
        exc = (res.get("exception_info") or {}).get("exception_type")
        t0, t1 = ts(res.get("started_at")), ts(res.get("finished_at"))
        dur = (t1 - t0) if t0 and t1 else None
        health.append((t1 or 0, exc not in OK_EXC or (dur is not None and dur < 120)))
        g = os.path.join(d, "verifier", "grade.json")
        if exc not in OK_EXC or not os.path.exists(g):
            continue
        grade = json.load(open(g))
        lower = bool(grade.get("is_lower_better"))
        b = board(a.tasks_dir, task)
        held = grade.get("heldout_score", grade.get("score")) if grade.get("heldout_valid", grade.get("valid_submission")) else None
        try:
            log = open(os.path.join(d, "agent", "steps.jsonl"), "rb").read()
        except OSError:
            log = b""
        scores = [float(x) for x in dict.fromkeys(SCORE_RE.findall(log))]
        rows.append(dict(v=start_version(spool, trial), task=task, test_q=q(b, held, lower),
                         first_q=q(b, scores[0], lower) if scores else 0.0, has_val=bool(scores),
                         valid=bool(grade.get("valid_submission")),
                         limit=exc in LIMIT_EXC or b"Agent reached maximum" in log[-200_000:],
                         dur=dur / 3600 if dur else None, vals=len(scores)))
    bins = defaultdict(list)
    for x in rows:
        if x["v"] is not None:
            bins[x["v"] // a.bin * a.bin].append(x)
    print(f"{'versions':>9} {'n':>4} {'tasks':>5} {'test_q':>7} {'first_q':>7} {'valid':>6} {'limit':>6} {'dur_h':>6} {'vals':>5}")
    for k in sorted(bins):
        b = bins[k]; m = lambda key: sum(x[key] for x in b) / len(b)
        durs = [x["dur"] for x in b if x["dur"] is not None]
        print(f"{k:>4}-{k + a.bin - 1:<4} {len(b):>4} {len({x['task'] for x in b}):>5} {m('test_q'):>7.3f} "
              f"{m('first_q'):>7.3f} {m('valid'):>6.2f} {m('limit'):>6.2f} "
              f"{st.median(durs) if durs else float('nan'):>6.2f} {st.median(x['vals'] for x in b):>5.0f}")
    print(f"(binned by the weight version a trial started on; test_q has a standard error of about "
          f"{0.3 / max(1, st.median(len(b) for b in bins.values()) if bins else 1) ** 0.5:.2f} per bin)")
    now = time.time(); since = now - a.hours * 3600
    recent = [x for x in health if x[0] >= since]
    fails = sum(1 for x in recent if x[1])
    sp = [os.path.getmtime(f) for f in glob.glob(os.path.join(spool, "*.jsonl"))]
    newest = (now - max(sp)) / 60 if sp else float("nan")
    print(f"last {a.hours:g}h: {len(recent) - fails} trials finished ({(len(recent) - fails) / a.hours:.1f}/h), "
          f"{fails} launch failures; newest spool write {newest:.0f} min ago")
    if recent and fails > 0.5 * len(recent):
        print("WARNING: most recent trials failed at launch (SAIL credits? relay?) - the trainer is starving")


if __name__ == "__main__":
    main()
