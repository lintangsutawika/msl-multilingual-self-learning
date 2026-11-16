"""MLE-Bench multilingual benchmark: generate Harbor tasks from a shared template."""

import os as _os

if _os.environ.get("MLEBENCH_DATA"):  # the data dir (scripts/prepare_mlebench.sh --data); MLE-bench's default otherwise
    from pathlib import Path as _Path

    import mlebench.registry as _r

    _r.registry = _r.registry.set_data_dir(_Path(_os.environ["MLEBENCH_DATA"]))

from .main import main  # noqa: E402

__all__ = ["main"]
