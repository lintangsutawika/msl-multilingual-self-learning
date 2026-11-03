"""SandMLE source: synthetic MLE tasks, each shipped with the generator script that made its data.

As distributed, a SandMLE task has ~30 test rows, too few for a public/private split. Every task ships
generate_task_env.py (pandas/numpy/scipy/PIL; writes into the working directory only; fixed seeds), with the
dataset size hardcoded as 147/137/127 samples. `build` rewrites those constants to `scale` times larger, reruns
the generator in an empty directory, and imports the result in MLE-bench layout with the task's own evaluator.py
as the grader. Generators without that constant (or that fail, time out, or need missing packages) are recorded
in <root>/_sandmle_status.tsv and skipped.

Caveats (see README): the labels are noisy draws, so a perfect score is not reachable (reward is normalized
between the sample submission and the true answers); SandMLE's medal thresholds are hand-written, not used.

download: the full archive is a 1.68 TiB .tar.gz on Google Drive; `download` streams it with HTTP Range requests
          (plain downloads hit Drive's quota page) and keeps the first N tasks (tar order, i.e. arbitrary).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

from common import GRADER_SRC, NEVER_PUBLIC, dir_mb, find_sample, is_mlebench, write_config

URL = "https://drive.usercontent.google.com/download?id=1qARX8hObQfLw3eXxpea79JOz2gjPAGmk&export=download&confirm=t"
CHUNK = 64 << 20
SIZE_CONST = re.compile(r"(?<![\w.])(147|137|127)(?![\w.])")


class _RangeReader:
    """File-like object over the Drive archive, fetched in byte ranges."""

    def __init__(self):
        self.n, self.off, self.buf = 0, 0, b""

    def _fetch(self) -> None:
        for attempt in range(8):
            try:
                req = urllib.request.Request(URL, headers={"Range": f"bytes={self.off}-{self.off + CHUNK - 1}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise RuntimeError(f"HTTP {r.status} {r.headers.get('content-type')}")
                    data = r.read()
                self.buf += data
                self.off += len(data)
                return
            except Exception as e:
                print(f"range {self.off}: {e} (retry {attempt})", flush=True)
                time.sleep(15 * (attempt + 1))
        raise RuntimeError("giving up on range fetch")

    def read(self, size=-1):
        while size < 0 or len(self.buf) < size:
            before = len(self.buf)
            self._fetch()
            if len(self.buf) == before or size < 0:
                break
        b, self.buf = (self.buf, b"") if size < 0 else (self.buf[:size], self.buf[size:])
        self.n += len(b)
        return b


def download(out: Path, n_tasks: int, max_gb: float) -> None:
    out.mkdir(parents=True, exist_ok=True)
    src, tasks = _RangeReader(), []
    with open(out / "_members.tsv", "w") as log, tarfile.open(fileobj=src, mode="r|gz") as tar:
        log.write("task\tpath\tsize\n")
        for m in tar:
            p = PurePosixPath(m.name)
            if p.is_absolute() or ".." in p.parts or not (m.isfile() or m.isdir()):
                continue  # untrusted archive: plain files and dirs inside the tree only
            task = p.parts[1] if len(p.parts) > 1 else ""
            if task and task not in tasks and (len(p.parts) > 2 or m.isdir()):
                if len(tasks) >= n_tasks:
                    break
                tasks.append(task)
                print(f"[{len(tasks)}] {task}  ({src.n / 1e9:.1f} GB streamed)", flush=True)
            log.write(f"{task}\t{m.name}\t{m.size}\n")
            if m.isfile():
                dest = out / Path(*p.parts[1:]) if len(p.parts) > 1 else out / p.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as fsrc, open(dest, "wb") as fdst:
                    while chunk := fsrc.read(1 << 20):
                        fdst.write(chunk)
            if src.n > max_gb * 1e9:
                print(f"stop: {max_gb} GB cap (last task may be partial)", flush=True)
                break
    print("tasks:", len(tasks))


def _higher_is_better(evaluator: Path) -> bool | None:
    m = re.search(r"^\s*(IS_LOWER_BETTER|LOWER_IS_BETTER|HIGHER_IS_BETTER)\s*=\s*(True|False)", evaluator.read_text(), re.M)
    if not m:
        return None
    return (m.group(2) == "True") == (m.group(1) == "HIGHER_IS_BETTER")


def _build_one(args) -> tuple[str, str, str]:
    src_pub, root, scale, timeout = args
    name = src_pub.parent.parent.name
    try:
        gen_src = (src_pub / "generate_task_env.py").read_text()
        hib = _higher_is_better(src_pub / "evaluator.py")
        if hib is None:
            return name, "no_direction", "evaluator does not declare its metric direction"
        gen, n_sub = SIZE_CONST.subn(lambda m: str(int(m.group(1)) * scale), gen_src)
        if n_sub == 0:
            return name, "no_size_knob", "no 147/137/127 size constant in generator"
        work = root / "_work" / name
        shutil.rmtree(work, ignore_errors=True)
        (work / "out").mkdir(parents=True)
        (work / "generate_task_env.py").write_text(gen)
        t0 = time.time()
        # -I: isolated mode, so nothing in the output dir can shadow an import.
        r = subprocess.run([sys.executable, "-I", str(work / "generate_task_env.py")], cwd=work / "out",
                           capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        (work / "generate.log").write_text(r.stdout[-20000:] + "\n--- stderr ---\n" + r.stderr[-20000:])
        if r.returncode != 0:
            return name, "generator_error", r.stderr.strip().splitlines()[-1][:200] if r.stderr.strip() else f"rc={r.returncode}"
        out = work / "out"
        if not (out / "answer.csv").exists():
            return name, "no_answer_csv", ""
        t = root / name
        shutil.rmtree(t, ignore_errors=True)
        (t / "prepared/private").mkdir(parents=True)
        shutil.move(str(out / "answer.csv"), t / "prepared/private/answers.csv")
        if (out / "threshold.json").exists():  # kept for reference only (hand-written medal thresholds)
            shutil.move(str(out / "threshold.json"), t / "prepared/private/threshold.json")
        for p in list(out.iterdir()):
            if p.name in NEVER_PUBLIC:
                p.unlink() if p.is_file() else shutil.rmtree(p)
        shutil.move(str(out), t / "prepared/public")
        if not (t / "prepared/public/description.md").exists():
            shutil.copy(src_pub / "description.md", t / "prepared/public/description.md")
        sample = find_sample(t / "prepared/public")
        if not sample:
            return name, "no_sample_submission", ""
        (t / "grader").mkdir()
        shutil.copy(src_pub / "evaluator.py", t / "grader/evaluator.py")
        (t / "grader/grade.py").write_text(
            (GRADER_SRC / "sandmle_grade.py").read_text().replace("__HIGHER_IS_BETTER__", str(hib)))
        write_config(t, {
            "id": name,
            "name": name,
            "source": "sandmle",
            "origin": "SandMLE (synthetic), regenerated",
            "sample_submission": sample.name,
            "id_column": None,
            "leaderboard": None,
            "reward": "normalized",
            "generator": {"scale": scale, "size_constants_replaced": n_sub, "seconds": round(time.time() - t0)},
        })
        shutil.rmtree(work / "out", ignore_errors=True)
        return name, "ok", f"{dir_mb(t / 'prepared/public')} MB public"
    except subprocess.TimeoutExpired:
        return name, "timeout", f">{timeout}s"
    except Exception as e:
        return name, "error", f"{type(e).__name__}: {e}"[:200]


def build(sample: Path, root: Path, scale: int, max_sample_mb: float, timeout: int, workers: int,
          only: list[str] | None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    todo, rows = [], []
    for pub in sorted(sample.glob("*/prepared/public")):
        name = pub.parent.parent.name
        if only and name not in only:
            continue
        if is_mlebench(name):
            rows.append((name, "mlebench", "MLE-bench competition (eval set)"))
            continue
        if not (pub / "generate_task_env.py").exists() or not (pub / "evaluator.py").exists():
            rows.append((name, "incomplete", "no generator or evaluator"))
        elif dir_mb(pub) > max_sample_mb:
            rows.append((name, "too_large", f"{dir_mb(pub)} MB as shipped; x{scale} would be too big"))
        else:
            todo.append((pub, root, scale, timeout))
    with ProcessPoolExecutor(workers) as ex:
        for row in ex.map(_build_one, todo):
            print("\t".join(row), flush=True)
            rows.append(row)
    st = root / "_sandmle_status.tsv"
    old = {l.split("\t")[0]: l for l in st.read_text().splitlines()[1:]} if st.exists() else {}
    old.update({r[0]: "\t".join(r) for r in rows})
    st.write_text("task\tstatus\tnote\n" + "\n".join(old[k] for k in sorted(old)) + "\n")
