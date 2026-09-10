from __future__ import annotations

import json
import subprocess
from abc import ABC
from pathlib import Path
from typing import Any

from .base import Executor


class NativeProcessExecutor(
    Executor,
    ABC,
):
    def __init__(
        self,
        workspace: Path,
        interface,
    ) -> None:
        super().__init__(
            workspace=workspace,
            interface=interface,
        )

        self.process: subprocess.Popen[str] | None = None
        self.stderr_file = None

    def _start_process(
        self,
        command: list[str],
    ) -> None:
        stderr_path = (
            self.workspace
            / "native-runner-stderr.txt"
        )

        self.stderr_file = stderr_path.open(
            "w",
            encoding="utf-8",
        )

        self.process = subprocess.Popen(
            command,
            cwd=self.workspace,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_file,
            text=True,
            bufsize=1,
        )

    def _jsonable(
        self,
        value: Any,
    ) -> Any:
        if value is None:
            return None

        if isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

        if isinstance(
            value,
            (list, tuple),
        ):
            return [
                self._jsonable(item)
                for item in value
            ]

        if isinstance(
            value,
            dict,
        ):
            return {
                str(key): self._jsonable(item)
                for key, item in value.items()
            }

        # LeetCode TreeNode-like object.
        if hasattr(value, "val"):
            return {
                "val": self._jsonable(
                    value.val
                ),
                "left": self._jsonable(
                    getattr(
                        value,
                        "left",
                        None,
                    )
                ),
                "right": self._jsonable(
                    getattr(
                        value,
                        "right",
                        None,
                    )
                ),
            }

        raise TypeError(
            f"Cannot serialize argument "
            f"of type {type(value)!r}"
        )

    def call(
        self,
        *args,
    ) -> Any:
        if self.process is None:
            raise RuntimeError(
                "Native runner is not prepared"
            )

        if (
            self.process.stdin is None
            or self.process.stdout is None
        ):
            raise RuntimeError(
                "Native runner pipes are unavailable"
            )

        request = {
            "args": [
                self._jsonable(arg)
                for arg in args
            ]
        }

        self.process.stdin.write(
            json.dumps(
                request,
                ensure_ascii=False,
            )
            + "\n"
        )
        self.process.stdin.flush()

        line = self.process.stdout.readline()

        if not line:
            return_code = self.process.poll()

            raise RuntimeError(
                "Native runner terminated "
                "without returning a result "
                f"(return code: {return_code})"
            )

        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Native runner returned "
                f"invalid JSON: {line!r}"
            ) from exc

        if not isinstance(
            response,
            dict,
        ):
            raise RuntimeError(
                "Native runner response "
                "must be a JSON object"
            )

        if not response.get(
            "ok",
            False,
        ):
            raise RuntimeError(
                "Native candidate failed: "
                f"{response.get('error', 'unknown error')}"
            )

        return response.get(
            "result"
        )

    def close(
        self,
    ) -> None:
        if self.process is not None:
            if self.process.stdin is not None:
                try:
                    self.process.stdin.close()
                except Exception:
                    pass

            try:
                self.process.terminate()
                self.process.wait(
                    timeout=2
                )
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass

            self.process = None

        if self.stderr_file is not None:
            try:
                self.stderr_file.close()
            except Exception:
                pass

            self.stderr_file = None