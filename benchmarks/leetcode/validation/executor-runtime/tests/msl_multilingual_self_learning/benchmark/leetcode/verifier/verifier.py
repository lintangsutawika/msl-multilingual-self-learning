from __future__ import annotations

import json
from pathlib import Path

from ..executors.base import Executor
from ..executors.cpp import CppExecutor
from ..executors.go import GoExecutor
from ..executors.java import JavaExecutor
from ..executors.python import PythonExecutor
from ..interfaces.models import SolutionInterface


EXECUTORS: dict[str, type[Executor]] = {
    "python": PythonExecutor,
    "cpp": CppExecutor,
    "go": GoExecutor,
    "java": JavaExecutor,
}


class SolutionVerifier:
    def __init__(
        self,
        workspace: Path,
        interface: SolutionInterface,
    ) -> None:
        self.workspace = workspace
        self.interface = interface
        self.executor: Executor | None = None

    def _load_solution_json(
        self,
    ) -> dict:
        solution_path = (
            self.workspace
            / "solution.json"
        )

        if not solution_path.exists():
            raise RuntimeError(
                "Missing candidate output: "
                "/workspace/solution.json"
            )

        try:
            raw = solution_path.read_text(
                encoding="utf-8"
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to read "
                "/workspace/solution.json"
            ) from exc

        try:
            solution = json.loads(
                raw
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Candidate output is not "
                "valid JSON"
            ) from exc

        if not isinstance(
            solution,
            dict,
        ):
            raise RuntimeError(
                "solution.json must contain "
                "a JSON object"
            )

        return solution

    def _validate_solution(
        self,
        solution: dict,
    ) -> tuple[str, str]:
        language = solution.get(
            "language"
        )

        code = solution.get(
            "code"
        )

        if not isinstance(
            language,
            str,
        ):
            raise RuntimeError(
                "solution.json field "
                "'language' must be a string"
            )

        if language != self.interface.language:
            raise RuntimeError(
                "Candidate language does not "
                "match task language: "
                f"expected "
                f"{self.interface.language!r}, "
                f"got {language!r}"
            )

        if not isinstance(
            code,
            str,
        ):
            raise RuntimeError(
                "solution.json field "
                "'code' must be a string"
            )

        code = code.strip()

        if not code:
            raise RuntimeError(
                "Candidate source code is empty"
            )

        return language, code

    def prepare(
        self,
    ) -> None:
        if self.executor is not None:
            raise RuntimeError(
                "Verifier has already "
                "been prepared"
            )

        solution = (
            self._load_solution_json()
        )

        language, code = (
            self._validate_solution(
                solution
            )
        )

        executor_class = (
            EXECUTORS.get(
                language
            )
        )

        if executor_class is None:
            supported = ", ".join(
                sorted(
                    EXECUTORS
                )
            )

            raise RuntimeError(
                "Unsupported candidate "
                f"language: {language!r}. "
                f"Supported languages: "
                f"{supported}"
            )

        executor = executor_class(
            workspace=self.workspace,
            interface=self.interface,
        )

        try:
            executor.prepare(
                code
            )
        except Exception:
            try:
                executor.close()
            except Exception:
                pass

            raise

        self.executor = executor

    def call(
        self,
        *args,
    ):
        if self.executor is None:
            raise RuntimeError(
                "Verifier has not been "
                "prepared"
            )

        return self.executor.call(
            *args
        )

    def close(
        self,
    ) -> None:
        if self.executor is None:
            return

        executor = self.executor
        self.executor = None

        executor.close()