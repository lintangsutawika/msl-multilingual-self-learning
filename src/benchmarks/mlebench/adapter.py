"""Adapter: generate Harbor MLE-Bench tasks from a single shared task template.

Mirrors src/benchmarks/leetcode/adapter.py: each competition is instantiated by
COPYING the per-benchmark task template (``task-template/``) into a fresh task
dir and substituting ``{placeholder}`` tokens with competition-specific values.

Unlike leetcode (which has one template per language), MLE-Bench has ONE template
shared by all competitions: the environment/Dockerfile installs mlebench and bakes the
prepared data in, the verifier runs ``mlebench grade``, and per-competition values (id,
name, difficulty, problem, grading metric) are filled in at generation.

Generated tasks land under ``<output>/<competition_id>/``.
"""

from __future__ import annotations

import json
import secrets
import shutil
import urllib.request
import zipfile
from pathlib import Path

from mlebench.data import download_and_prepare_dataset, is_dataset_prepared
from mlebench.registry import registry

# Template package dir (this file lives in src/benchmarks/mlebench/adapter.py).
PKG = Path(__file__).parent
TEMPLATE = PKG / "task-template"
# Same rev as the mlebench pin in pyproject.toml. Upstream leaderboard.csv files are Git LFS
# objects (a plain clone or pip install only has pointers), so they are downloaded from here.
MLEBENCH_REV = "507f92e1138bb6e40dac5c6ee7a6758e6424bf97"
LEADERBOARD_URL = ("https://media.githubusercontent.com/media/openai/mle-bench/"
                   + MLEBENCH_REV + "/mlebench/competitions/{cid}/leaderboard.csv")


def _fill(path: Path, **kw: str) -> None:
    """Substitute ``{key}`` placeholders in a file in place (single pass)."""
    if not path.exists():
        return
    text = path.read_text()
    for k, v in kw.items():
        text = text.replace("{" + k + "}", str(v))
    path.write_text(text)


def _metric_notes(comp) -> str:
    grader_name = getattr(getattr(comp, "grader", None), "name", None)
    ct = getattr(comp, "competition_type", "")
    return f"metric={grader_name or 'unknown'} type={ct}"


def _problem_desc(comp) -> str:
    """Return the competition's SANITIZED problem statement for the agent.

    mlebench ships both description.md (full, may leak raw feature/test-set
    details) and description_obfuscated.md (the agent-facing sanitized prompt).
    Always use the obfuscated one; fall back to the full description if a
    competition has no obfuscated file.
    """
    try:
        obf = (
            registry.get_competitions_dir()
            / getattr(comp, "id", comp) / "description_obfuscated.md"
        )
        if obf.exists():
            return obf.read_text().strip()[:2000]
    except Exception:
        pass
    return str(getattr(comp, "description", "") or "").strip()[:2000]


def prepare(competition_id: str) -> None:
    """Download and prepare a competition's data into the mlebench cache (needs Kaggle
    credentials). No-op if already prepared. Kaggle re-uploads change file checksums, so
    they are overwritten; a corrupt (half-downloaded) zip is removed and retried once."""
    comp = registry.get_competition(competition_id)
    if is_dataset_prepared(comp):
        return
    from .valsplit import require_pandas2
    require_pandas2()
    zip_path = comp.raw_dir.parent / f"{competition_id}.zip"
    _drop_partial_extraction(comp, zip_path)
    for attempt in (1, 2):
        if zip_path.exists() and not zipfile.is_zipfile(zip_path):
            zip_path.unlink()
        try:
            download_and_prepare_dataset(comp, overwrite_checksums=True)
            return
        except zipfile.BadZipFile:
            zip_path.unlink(missing_ok=True)
            if attempt == 2:
                raise


def _drop_partial_extraction(comp, zip_path: Path) -> None:
    """MLE-bench extracts the zip only into an EMPTY raw/: an extraction cut short (preemption) leaves raw/ partial
    for good, and prepare.py then fails on a missing file (AI4Code: train_ancestors.csv). If raw/ lacks any member of
    the zip, remove it and the half-written prepared/ so the next prepare extracts again."""
    raw = comp.raw_dir
    if not (raw.is_dir() and any(raw.iterdir()) and zipfile.is_zipfile(zip_path)):
        return
    with zipfile.ZipFile(zip_path) as z:
        missing = [n for n in z.namelist() if not n.endswith("/") and not (raw / n).exists()]
    if missing:
        print(f"{comp.id}: raw/ is missing {len(missing)} of the zip's files (e.g. {missing[0]}): extracting again",
              flush=True)
        shutil.rmtree(raw)
        for d in (comp.public_dir, comp.private_dir):
            shutil.rmtree(d, ignore_errors=True)


def _leaderboard(comp) -> Path:
    """The competition's real Kaggle leaderboard (medal thresholds), cached next to its data."""
    path = registry.get_data_dir() / comp.id / "leaderboard.csv"
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(LEADERBOARD_URL.format(cid=comp.id), tmp)
        if tmp.read_bytes().startswith(b"version https://git-lfs"):
            raise RuntimeError(f"{comp.id}: got an LFS pointer instead of the leaderboard")
        tmp.replace(path)
    return path


# Lines of the template Dockerfile that bake the data in; tasks built with mount_data drop them.
BAKED_DATA_LINES = ("COPY public/ /workspace/public/", "COPY private/ /workspace/private/",
                    "COPY leaderboard.csv /workspace/leaderboard.csv")


def _bake(task: Path, comp, val_split: bool = False, mount_data: bool = False) -> None:
    """Copy the prepared public/private splits and the leaderboard into environment/,
    where the Dockerfile COPYs them into the image (no Kaggle access at runtime).
    With val_split, public/ is the validation-split public (MLE-bench's test inputs + validation inputs) and
    private_val/ holds the validation answers (valsplit.py); private/ is MLE-bench's test answers either way."""
    env = task / "environment"
    public = comp.public_dir
    if val_split:
        from .valsplit import make

        vs = make(comp.id)
        public = vs / "public"
        if mount_data:  # the answers stay on the host (validate_host.py grades there); only the marker goes in
            (env / "private_val").mkdir()
        else:
            shutil.copytree(vs / "private_val", env / "private_val")
        shutil.copy(vs / "valsplit.json", env / "private_val" / "valsplit.json")  # turns validation mode on
    else:
        (env / "private_val").mkdir()
        (env / "private_val" / "NONE").write_text("no validation split: validate scores MLE-bench's test answers\n")
    if mount_data:
        # The public data stays where it is and is bind-mounted read-only at /workspace/public at run time (its
        # path is in task.toml [metadata] public_data); the test answers and leaderboard go to tests/, which Harbor
        # uploads only for the final grade (the agent never sees them). Only small files are copied.
        shutil.copytree(comp.private_dir, task / "tests" / "private")
        shutil.copy(_leaderboard(comp), task / "tests" / "leaderboard.csv")
        df = env / "Dockerfile"
        lines = [l for l in df.read_text().splitlines(keepends=True) if l.strip() not in BAKED_DATA_LINES]
        # Rootless Singularity can't keep a root-only file from the agent: the signing key stays on the host too
        # (environment/score_key, read by validate_host.py and the rollout forwarder), and the image gets an empty one.
        lines = ["RUN mkdir -p /etc/msl && : > /etc/msl/score_key\n" if l.strip() == "COPY score_key /etc/msl/score_key"
                 else l for l in lines]
        df.write_text("".join(lines))
        toml = task / "task.toml"
        meta = f"[metadata]\npublic_data = {json.dumps(str(public.resolve()))}  # bind-mount at /workspace/public\n"
        if val_split:
            meta += f"val_answers = {json.dumps(str((vs / 'private_val').resolve()))}  # graded on the host (validate_host.py)\n"
        text = toml.read_text().replace("[metadata]\n", meta, 1)
        # Rootless Singularity fakes uids: a non-root agent's own files show up as root's, and tools that check
        # ownership (tmux, ssh) refuse them. The agent user isolates nothing there (answers and key are on the host).
        assert '\nuser = "agent"\n' in text
        text = text.replace('\nuser = "agent"\n', '\nuser = "root"  # Babel/Singularity: see adapter._bake\n', 1)
        # Babel sweeps give each attempt what the MLE-Dojo sweep's SAIL sandboxes had: 4 CPUs, 16 GB. Singularity
        # enforces only the memory (Harbor kills the trial above it); the CPUs are the Slurm job's, shared.
        for k, v in (("cpus", "4"), ("memory_mb", "16384")):
            assert f"\n{k} = " in text
            text = "\n".join(f"{k} = {v}" if l.startswith(f"{k} = ") else l for l in text.split("\n"))
        # The verifier grades from tests/ with no Kaggle or SAIL access, so it gets no credentials.
        toml.write_text("".join(l for l in text.splitlines(keepends=True) if not l.startswith("env = { KAGGLE_USERNAME")))
        return
    for src, name in ((public, "public"), (comp.private_dir, "private")):
        shutil.copytree(src, env / name)
    shutil.copy(_leaderboard(comp), env / "leaderboard.csv")


def generate(competition_id: str, difficulty: str, output: Path, val_split: bool = False,
             mount_data: bool = False) -> Path:
    """Instantiate one competition from the shared template into ``output``, with its
    prepared data baked in (run ``prepare`` first)."""
    comp = registry.get_competition(competition_id)
    if not is_dataset_prepared(comp):
        raise RuntimeError(f"{competition_id}: data not prepared; rerun with --prepare (needs Kaggle credentials)")
    task = output / competition_id
    # Keep an existing score-signing key: the rollout forwarder verifies `validate` scores
    # with it, so a new key would reject scores from images built before regeneration.
    key = task / "environment" / "score_key"
    old_key = key.read_text() if key.exists() else None
    if task.exists():
        # Re-generate idempotently: wipe stale copies so fills are clean.
        shutil.rmtree(task)
    shutil.copytree(TEMPLATE, task, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    valsplit_instruction = task / "instruction-valsplit.md"
    if val_split:
        valsplit_instruction.replace(task / "instruction.md")
    else:
        valsplit_instruction.unlink()

    fills = {
        "comp_id": competition_id,
        "comp_name": str(getattr(comp, "name", competition_id) or competition_id),
        "difficulty": difficulty,
        "metric_notes": _metric_notes(comp),
        "description": _problem_desc(comp),
        "mlebench_rev": MLEBENCH_REV,
    }
    for path in (
        task / "task.toml",
        task / "instruction.md",
        task / "environment" / "Dockerfile",
        task / "environment" / "validate_server.py",
        task / "environment" / "validate.py",
        task / "environment" / "startup.sh",
        task / "tests" / "test.sh",
    ):
        _fill(path, **fills)
    _bake(task, comp, val_split, mount_data)
    # Per-task score-signing key: copied root-only into the image; the rollout forwarder
    # reads it (SCORE_KEYS_DIR/<comp>/environment/score_key) to verify `validate` scores.
    key.write_text(old_key or secrets.token_hex(32) + "\n")
    key.chmod(0o600)
    for _bin in ("tests/test.sh", "solution/solve.sh", "environment/startup.sh", "environment/bin/validate",
                 "environment/bin/validate-root"):
        _p = task / _bin
        if _p.exists():
            _p.chmod(0o755)
    return task


def generate_all(
    targets: dict[str, str],
    output: Path,
    prepare_data: bool = False,
    val_split: bool = False,
    mount_data: bool = False,
) -> tuple[list[tuple[str, str, Path]], list[tuple[str, str]]]:
    """Generate all (competition_id -> difficulty) targets under output.
    Returns (made, failed); one competition failing does not stop the rest."""
    output.mkdir(parents=True, exist_ok=True)
    made: list[tuple[str, str, Path]] = []
    failed: list[tuple[str, str]] = []
    for cid, diff in sorted(targets.items()):
        split_out = output / f"mlebench-{diff}"
        split_out.mkdir(parents=True, exist_ok=True)
        try:
            if prepare_data:
                prepare(cid)
            made.append((cid, diff, generate(cid, diff, split_out, val_split, mount_data)))
        except Exception as e:
            failed.append((cid, f"{type(e).__name__}: {e}"))
    return made, failed
