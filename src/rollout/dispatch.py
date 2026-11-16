"""Rollout dispatcher: run real MLE-Bench trials against this rollout job's relay.

Each trial is one `harbor run` of the openhands-sdk agent (with a history condenser,
src/rollout/openhands_agent.py) in a SAIL sandbox,
with the agent's LLM pointed at the relay *session* URL for that trial:

    LLM_BASE_URL = <relay>/sessions/<trial>/v1

so the rollout job's forwarder files every call of the trial under spool/<trial>.jsonl.
The relay URL/secret come from the rollout job's endpoint.env (fresh per rollout job).

Runs the task list round-robin (--rounds passes, 0 = forever; --shuffle reorders each pass) with
bounded concurrency. New trials are held while the relay is down (its /health fails or reports no
bridge), and resume when it is back. A run of trials that all end within --fast-fail-secs of launch
(e.g. SAIL refusing sandboxes: on 2026-11-09 an exhausted credit balance failed 6065 trials in 7h
while the trainer kept stepping on the few long trials still alive) pauses new trials for
--fast-fail-pause seconds, then probes again. Later: take tasks from the
task_queue/ the trainer writes, and honor dispatch_ctl.json (pause / max in flight).

    python src/rollout/dispatch.py --run-dir <MSL_STORE>/rollout --tasks aerial-cactus-identification
"""
import argparse
import json
import os
import random
import urllib.request
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def read_env(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def log(msg: str) -> None:
    print(f"[dispatch {datetime.now():%H:%M:%S}] {msg}", flush=True)


def relay_ok(ep: dict) -> bool:
    """True if the relay answers /health with the node's bridge connected. When Modal
    recycles the relay container and cannot reschedule it, every LLM call times out and
    a trial started then just burns a sandbox and fails."""
    try:
        with urllib.request.urlopen(f"{ep['RELAY_URL'].rstrip('/')}/health", timeout=20) as r:
            return bool(json.loads(r.read()).get("bridge_connected"))
    except Exception:
        return False


_VAL_STATE: dict[str, tuple[int, set, int]] = {}  # spool file -> (bytes read, raw scores seen, fresh results)


def fresh_validates(spool: Path) -> int:
    """Fresh validate results recorded for a trial so far (a score shown again is not a new result)."""
    offset, seen, n = _VAL_STATE.get(str(spool), (0, set(), 0))
    try:
        size = spool.stat().st_size
    except OSError:
        return n
    if size > offset:
        with open(spool, "rb") as fh:
            fh.seek(offset)
            chunk = fh.read()
        end = chunk.rfind(b"\n") + 1
        for line in chunk[:end].splitlines():
            if b"observed_validate_scores" not in line or b'"observed_validate_scores": []' in line:
                continue
            try:
                raw = json.loads(line).get("observed_validate_scores") or []
            except Exception:
                continue
            if raw and not all(x in seen for x in raw):
                seen.update(raw)
                n += 1
        _VAL_STATE[str(spool)] = (offset + end, seen, n)
    return n


def launch(task: str, trial: str, ep: dict, a, logs: Path) -> subprocess.Popen:
    base = f"{ep['RELAY_URL'].rstrip('/')}/sessions/{trial}/v1"
    env = dict(os.environ)
    # Task verifiers need KAGGLE_USERNAME/KAGGLE_KEY and SAIL_API_KEY; take them from the
    # repo .env (never printed). Values already in the environment win.
    for k, v in read_env(REPO / ".env").items():
        env.setdefault(k, v)
    env["SAIL_API_KEY"] = env.get("SAIL_API_KEY") or env.get("SAIL_KEY", "")
    env["PYTHONPATH"] = f"{REPO}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(REPO)  # -a src.rollout...
    cmd = [
        str(REPO / ".venv" / "bin" / "harbor"), "run",
        "-p", str(REPO / a.tasks_dir / task),
        "-a", a.agent, "-m", f"litellm_proxy/{a.model}",
        "-e", "sail.harbor:SailboxEnvironment",
        "-n", "1",
        "--ae", f"LLM_API_KEY={ep['RELAY_SECRET']}",
        "--ae", f"LLM_BASE_URL={base}",
        "--ae", f"LLM_MODEL=litellm_proxy/{a.model}",
        "--ae", f"MAX_ITERATIONS={a.max_iterations}",
        *(["--ae", f"CONDENSER_MAX_EVENTS={a.condenser_max_events}"] if a.condenser_max_events else []),
        *(["--ae", f"TERMINAL_MAX_TIMEOUT={a.terminal_max_timeout}"] if a.terminal_max_timeout else []),
        *(["--ae", f"LLM_TIMEOUT={a.llm_timeout}"] if a.llm_timeout else []),
        *(["--agent-timeout-multiplier", str(a.agent_timeout_multiplier)] if a.agent_timeout_multiplier else []),
        "--job-name", trial,
        "-o", str(a.jobs_dir), "-y",
    ]
    out = open(logs / f"{trial}.log", "w")
    log(f"start {trial} (task {task}) -> {base}")
    return subprocess.Popen(["timeout", str(a.trial_timeout), *cmd], cwd=REPO, env=env,
                            stdout=out, stderr=subprocess.STDOUT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="the rollout job run dir (has endpoint.env)")
    ap.add_argument("--tasks", required=True, help='comma-separated task dirs under --tasks-dir, or "all"')
    ap.add_argument("--tasks-dir", default="harbor_tasks/mlebench-low")
    ap.add_argument("--model", default="qwen3.5-9B")
    ap.add_argument("--agent", default="src.rollout.openhands_agent:CondensingOpenHandsSDK",
                    help="harbor agent name or import path (stock: openhands-sdk)")
    ap.add_argument("--condenser-max-events", type=int, default=120,
                    help="summarize history past this many events (0 = never; needs the condensing agent)")
    ap.add_argument("--terminal-max-timeout", type=int, default=1800,
                    help="cap on how long one terminal action may block, seconds (0 = no cap; needs the condensing agent)")
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--max-iterations", type=int, default=200)
    ap.add_argument("--trial-timeout", type=int, default=14400,
                    help="outer kill of the whole harbor run, seconds (keep above the agent timeout + setup + verifier)")
    ap.add_argument("--agent-timeout-multiplier", type=float, default=0,
                    help="harbor multiplier on the task's [agent] timeout_sec (0 = harbor default 1; 4h tasks x3 = 12h)")
    ap.add_argument("--llm-timeout", type=int, default=0,
                    help="agent-side timeout of one model call, seconds (0 = the agent default, 60)")
    ap.add_argument("--shuffle", action="store_true", help="shuffle the task order on every pass")
    ap.add_argument("--seed", type=int, default=None, help="seed the task shuffle (default: unseeded)")
    ap.add_argument("--fast-fail-secs", type=int, default=120,
                    help="a trial ending this soon after launch counts as a launch failure")
    ap.add_argument("--fast-fail-run", type=int, default=8,
                    help="pause after this many launch failures in a row (0 = never)")
    ap.add_argument("--fast-fail-pause", type=int, default=600, help="seconds to hold new trials after such a run")
    ap.add_argument("--rounds", type=int, default=1, help="passes over the task list (0 = forever)")
    ap.add_argument("--stop-after-validates", type=int, default=0,
                    help="K > 0: stop a trial (SIGTERM, which harbor turns into a clean shutdown of its sandbox) once "
                         "its spool shows K fresh validate results. For runs that train only on episodes 1..K "
                         "(SPOOL_MAX_EPISODE=K): the rest of the trial would never be trained on. Default 0 = off.")
    a = ap.parse_args()
    # harbor uploads the agent's run_agent.py with the local file's mode, and the agent runs as an
    # unprivileged user in the sandbox: under ABCI's default umask 027 it could not read it and every
    # trial exited at once (exit 2, Permission denied; the 2026-11-08 baseline). rollout.sh happened
    # to set umask 022 before starting the dispatcher; set it here for every launcher.
    os.umask(0o022)

    run_dir = Path(a.run_dir)
    ep = read_env(run_dir / "endpoint.env")
    if not ep.get("RELAY_URL") or not ep.get("RELAY_SECRET"):
        log(f"no endpoint in {run_dir}/endpoint.env")
        return 1
    a.jobs_dir = run_dir / "trials"
    logs = run_dir / "dispatch_logs"
    a.jobs_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    if not (read_env(REPO / ".env").get("SAIL_KEY") or os.environ.get("SAIL_API_KEY")):
        log("no SAIL key (SAIL_KEY in .env or SAIL_API_KEY)")
        return 1

    # Only tasks with data baked into the image work offline: without environment/public
    # the sandbox has no /workspace/public and the agent burns its run hunting for data.
    tasks = []
    names = ([d.name for d in sorted((REPO / a.tasks_dir).iterdir()) if (d / "task.toml").is_file()]
             if a.tasks == "all" else [x for x in a.tasks.split(",") if x])
    for t in names:
        if (REPO / a.tasks_dir / t / "environment" / "public").is_dir():
            tasks.append(t)
        else:
            log(f"skip {t}: no baked data at {a.tasks_dir}/{t}/environment/public")
    if not tasks:
        log("no runnable tasks")
        return 1

    def trials():
        # Round-robin over the tasks; trial ids start with the task (= competition) id,
        # which the forwarder uses to pick the score-signing key.
        # A task listed twice (e.g. re-running lost trials) can start twice in the same second:
        # a repeated id gets a "d<k>" suffix so the two trials do not share a run dir / spool file.
        r = 0
        used: set[str] = set()
        while a.rounds == 0 or r < a.rounds:
            for t in (random.sample(tasks, len(tasks)) if a.shuffle else tasks):
                base = f"{t}__{datetime.now().strftime('%m%d%H%M%S')}r{r}"
                tid, k = base, 1
                while tid in used:
                    k += 1
                    tid = f"{base}d{k}"
                used.add(tid)
                yield t, tid
            r += 1

    pending = trials()
    if a.seed is not None:
        random.seed(a.seed)
    running: dict[str, subprocess.Popen] = {}
    early_stopped: set[str] = set()
    started: dict[str, float] = {}
    exhausted = False
    paused = False
    fast_fails, hold_until = 0, 0.0
    while not exhausted or running:
        if not exhausted and len(running) < a.max_concurrent:
            up = relay_ok(ep)
            if up == paused:
                paused = not up
                log("relay down (no /health or no bridge); pausing new trials" if paused else "relay back; resuming")
        # After a run of launch failures, start one probe trial per pause instead of filling every slot.
        held = time.time() < hold_until
        while not paused and not held and not exhausted and len(running) < a.max_concurrent:
            nxt = next(pending, None)
            if nxt is None:
                exhausted = True
                break
            running[nxt[1]] = launch(*nxt, ep, a, logs)
            started[nxt[1]] = time.time()
            if a.fast_fail_run and fast_fails >= a.fast_fail_run:
                break
        for trial, p in list(running.items()):
            if (a.stop_after_validates > 0 and trial not in early_stopped and p.poll() is None
                    and fresh_validates(Path(a.run_dir) / "spool" / f"{trial}.jsonl") >= a.stop_after_validates):
                p.send_signal(signal.SIGTERM)  # `timeout` relays it to harbor, which stops the sandbox
                early_stopped.add(trial)
                log(f"early stop {trial}: {a.stop_after_validates} validate results recorded")
            if p.poll() is not None:
                secs = time.time() - started.pop(trial)
                log(f"done {trial} exit={p.returncode} after {secs:.0f}s")
                del running[trial]
                if trial in early_stopped:
                    early_stopped.discard(trial)
                    continue  # stopped on purpose, not a launch failure
                if secs < a.fast_fail_secs:
                    fast_fails += 1
                    if a.fast_fail_run and fast_fails >= a.fast_fail_run:
                        hold_until = time.time() + a.fast_fail_pause
                        log(f"LAUNCH FAILURES: {fast_fails} trials in a row ended within {a.fast_fail_secs}s "
                            f"(see {logs}/{trial}.log); holding new trials for {a.fast_fail_pause}s, then probing")
                else:
                    if fast_fails >= a.fast_fail_run > 0:
                        log("launch failures cleared; resuming")
                    fast_fails = 0
        time.sleep(15)
    log("all trials finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
