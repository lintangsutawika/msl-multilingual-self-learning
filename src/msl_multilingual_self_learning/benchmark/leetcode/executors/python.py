from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from msl_multilingual_self_learning.benchmark.leetcode.executors.base import (
    Executor,
)


class PythonExecutor(Executor):
    def __init__(
        self,
        workspace: Path,
        interface,
    ) -> None:
        super().__init__(
            workspace=workspace,
            interface=interface,
        )

        self.solution_path = (
            self.workspace / "solution.py"
        )

        self.module = None
        self.candidate = None

    def prepare(
        self,
        code: str,
    ) -> None:
        """
        Write the candidate Python source and import it.
        """
        self.workspace.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.solution_path.write_text(
            code,
            encoding="utf-8",
        )

        module_name = (
            "_leetcode_candidate_solution"
        )

        spec = importlib.util.spec_from_file_location(
            module_name,
            self.solution_path,
        )

        if spec is None or spec.loader is None:
            raise RuntimeError(
                "Could not create import specification "
                f"for {self.solution_path}"
            )

        module = importlib.util.module_from_spec(
            spec
        )

        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)

        except Exception as exc:
            raise RuntimeError(
                "Failed to import candidate Python solution"
            ) from exc

        self.module = module

        self.candidate = self._resolve_candidate()

    def _resolve_candidate(
        self,
    ):
        if self.module is None:
            raise RuntimeError(
                "Executor has not been prepared"
            )

        container = self.interface.container
        callable_name = (
            self.interface.callable_name
        )

        if container is None:
            candidate = getattr(
                self.module,
                callable_name,
                None,
            )

            if candidate is None:
                raise RuntimeError(
                    f"Candidate does not define "
                    f"{callable_name!r}"
                )

            if not callable(candidate):
                raise RuntimeError(
                    f"{callable_name!r} "
                    "is not callable"
                )

            return candidate

        container_type = getattr(
            self.module,
            container,
            None,
        )

        if container_type is None:
            raise RuntimeError(
                f"Candidate does not define "
                f"container {container!r}"
            )

        try:
            instance = container_type()

        except Exception as exc:
            raise RuntimeError(
                f"Could not instantiate "
                f"{container!r}"
            ) from exc

        candidate = getattr(
            instance,
            callable_name,
            None,
        )

        if candidate is None:
            raise RuntimeError(
                f"{container!r} does not define "
                f"{callable_name!r}"
            )

        if not callable(candidate):
            raise RuntimeError(
                f"{container}.{callable_name} "
                "is not callable"
            )

        return candidate

    def call(
        self,
        *args: Any,
    ) -> Any:
        if self.candidate is None:
            raise RuntimeError(
                "Executor has not been prepared"
            )

        return self.candidate(
            *args
        )

    def close(
        self,
    ) -> None:
        self.module = None
        self.candidate = None