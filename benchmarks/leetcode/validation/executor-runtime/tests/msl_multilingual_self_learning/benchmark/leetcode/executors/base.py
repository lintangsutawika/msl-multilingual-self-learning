from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from msl_multilingual_self_learning.benchmark.leetcode.interfaces.models import (
    SolutionInterface,
)


class Executor(ABC):
    """
    Base interface for executing model-generated LeetCode solutions.

    An executor is responsible for:
      1. materializing source code
      2. preparing/compiling it
      3. invoking the required callable
      4. returning the result as a Python value
    """

    def __init__(
        self,
        workspace: Path,
        interface: SolutionInterface,
    ) -> None:
        self.workspace = workspace
        self.interface = interface

    @abstractmethod
    def prepare(
        self,
        code: str,
    ) -> None:
        """
        Write and, when necessary, compile the candidate code.

        Raises an exception if preparation fails.
        """
        raise NotImplementedError

    @abstractmethod
    def call(
        self,
        *args: Any,
    ) -> Any:
        """
        Invoke the candidate callable with Python arguments.

        Returns a Python-compatible result.
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Clean up any persistent child process.

        Executors that do not need cleanup can leave this unchanged.
        """
        return None