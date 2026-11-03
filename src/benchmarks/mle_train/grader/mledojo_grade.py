"""grade.py for tasks imported from MLE-Dojo: wraps the competition's CompetitionMetrics class (metric.py).

Copied into <task>/grader/ together with metric.py and MLE-Dojo's mledojo/metrics/base.py.
"""

import importlib.util
import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mledojo.metrics.base import CompetitionMetrics  # noqa: E402


def _load() -> CompetitionMetrics:
    spec = importlib.util.spec_from_file_location("dojo_metric", HERE / "metric.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if issubclass(obj, CompetitionMetrics) and obj is not CompetitionMetrics:
            return obj()
    raise RuntimeError("metric.py has no CompetitionMetrics subclass")


_METRIC = _load()
HIGHER_IS_BETTER = bool(_METRIC.higher_is_better)


def grade(submission, answers) -> float:
    # MLE-Dojo metrics raise InvalidSubmissionError; the base class also allows returning (False, reason).
    ok = _METRIC.validate_submission(submission.copy(), answers.copy())
    if isinstance(ok, tuple) and ok and ok[0] is False:
        raise ValueError(ok[1] if len(ok) > 1 else "invalid submission")
    return float(_METRIC.evaluate(answers.copy(), submission.copy()))
