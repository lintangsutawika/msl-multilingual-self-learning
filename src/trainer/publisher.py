"""Checkpoint publisher: MILES --custom-megatron-post-save-hook-path for the trainer.

MILES calls ``post_save(args, rollout_id, checkpoint_dir, hf_checkpoint_dir)`` on rank 0
after each save (once for the actor, with the HF export dir, and once for the critic,
with ``hf_checkpoint_dir=None``). For the actor it

  1. copies the base model's non-weight files (config, tokenizer, chat template, ...)
     into the HF export, which only holds weights + index, so SGLang can load it;
  2. atomically replaces ``<run>/LATEST`` with {"version": N, "path": ...},
     which the rollout job's weight watcher polls (version 0 = the base model; versions
     keep increasing across trainer restarts);
  3. prunes old HF exports, keeping the newest KEEP_HF (and never the one LATEST names).

For both roles it prunes old Megatron ``iter_*`` dirs, keeping the newest KEEP_CKPT
(an actor + critic save is ~260 GB).

Env:
    HF_BASE_DIR   base HF snapshot to copy non-weight files from (required)
    KEEP_HF       HF exports to keep (default 3)
    KEEP_CKPT     Megatron checkpoints to keep per role (default 1)
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")


def _log(msg: str) -> None:
    print(f"[publisher] {msg}", flush=True)


def _copy_base_files(hf: Path) -> None:
    base = Path(os.environ["HF_BASE_DIR"])
    for f in base.iterdir():
        if f.is_file() and not f.name.endswith(WEIGHT_SUFFIXES) and f.name != "model.safetensors.index.json":
            if not (hf / f.name).exists():
                shutil.copy(f.resolve(), hf / f.name)  # snapshot files are symlinks into blobs/


def _publish(run_dir: Path, version: int, hf: Path) -> None:
    tmp = run_dir / "LATEST.tmp"
    tmp.write_text(json.dumps({"version": version, "path": str(hf)}) + "\n")
    os.replace(tmp, run_dir / "LATEST")


def _prune(dirs: list[Path], keep: int, protect: set[Path]) -> None:
    for d in sorted(dirs, key=lambda p: p.stat().st_mtime)[:-keep or None]:
        if d.resolve() not in protect:
            shutil.rmtree(d, ignore_errors=True)
            _log(f"pruned {d}")


def post_save(args, rollout_id: int, checkpoint_dir, hf_checkpoint_dir) -> None:
    ckpt = Path(checkpoint_dir)
    _prune([d for d in ckpt.parent.glob("iter_*") if d != ckpt], int(os.environ.get("KEEP_CKPT", "1")) - 1, set())
    if hf_checkpoint_dir is None:
        return  # critic save
    hf = Path(hf_checkpoint_dir)
    if not any(hf.glob("*.safetensors")):
        _log(f"no weights in {hf}; not publishing")
        return
    _copy_base_files(hf)
    run_dir = Path(args.save).parent
    # Versions only increase, also across trainer restarts (a restart counts rollout_id from 0
    # again, and the weight watcher only swaps to a version newer than the served one).
    try:
        prev = int(json.loads((run_dir / "LATEST").read_text())["version"])
    except Exception:
        prev = 0
    version = max(rollout_id + 1, prev + 1)
    _publish(run_dir, version, hf)
    _log(f"published v{version} -> {hf}")
    _prune([d for d in hf.parent.glob("step_*") if d != hf], int(os.environ.get("KEEP_HF", "3")) - 1, {hf.resolve()})
