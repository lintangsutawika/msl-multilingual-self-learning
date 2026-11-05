"""Rollout dispatcher: run real MLE-Bench trials against this rollout job's relay.

Each trial is one `harbor run` of the openhands-sdk agent (with a history condenser,
src/rollout/openhands_agent.py) in a SAIL sandbox,
with the agent's LLM pointed at the relay *session* URL for that trial:

    LLM_BASE_URL = <relay>/sessions/<trial>/v1

so the rollout job's forwarder files every call of the trial under spool/<trial>.jsonl.
The relay URL/secret come from the rollout job's endpoint.env (fresh per rollout job).

Runs the task list round-robin (--rounds passes, 0 = forever) with bounded concurrency. Later: take tasks from the
task_queue/ the trainer writes, and honor dispatch_ctl.json (pause / max in flight).

    python src/rollout/dispatch.py --run-dir <MSL_STORE>/rollout --tasks aerial-cactus-identification
"""
import argparse
import os
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
    ap.add_argument("--tasks", required=True, help="comma-separated task dirs under --tasks-dir")
    ap.add_argument("--tasks-dir", default="harbor_tasks/mlebench-low")
    ap.add_argument("--model", default="qwen3.5-9B")
    ap.add_argument("--agent", default="src.rollout.openhands_agent:CondensingOpenHandsSDK",
                    help="harbor agent name or import path (stock: openhands-sdk)")
    ap.add_argument("--condenser-max-events", type=int, default=120,
                    help="summarize history past this many events (0 = never; needs the condensing agent)")
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--max-iterations", type=int, default=200)
    ap.add_argument("--trial-timeout", type=int, default=14400)
    ap.add_argument("--rounds", type=int, default=1, help="passes over the task list (0 = forever)")
    a = ap.parse_args()

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
    for t in (x for x in a.tasks.split(",") if x):
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
        r = 0
        while a.rounds == 0 or r < a.rounds:
            for t in tasks:
                yield t, f"{t}__{datetime.now().strftime('%m%d%H%M%S')}r{r}"
            r += 1

    pending = trials()
    running: dict[str, subprocess.Popen] = {}
    exhausted = False
    while not exhausted or running:
        while not exhausted and len(running) < a.max_concurrent:
            nxt = next(pending, None)
            if nxt is None:
                exhausted = True
                break
            running[nxt[1]] = launch(*nxt, ep, a, logs)
        for trial, p in list(running.items()):
            if p.poll() is not None:
                log(f"done {trial} exit={p.returncode}")
                del running[trial]
        time.sleep(15)
    log("all trials finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
