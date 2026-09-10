from __future__ import annotations

import subprocess
from pathlib import Path

from .native import NativeProcessExecutor


class GoExecutor(
    NativeProcessExecutor,
):
    def prepare(
        self,
        code: str,
    ) -> None:
        solution_path = (
            self.workspace
            / "solution.go"
        )

        runner_path = (
            self.workspace
            / "runner.go"
        )

        binary_path = (
            self.workspace
            / "runner"
        )

        code = code.strip()

        if not code.startswith(
            "package "
        ):
            code = (
                "package main\n\n"
                + code
            )

        solution_path.write_text(
            code + "\n",
            encoding="utf-8",
        )

        decode_lines: list[str] = []
        argument_names: list[str] = []

        for i, parameter in enumerate(
            self.interface.parameters
        ):
            name = f"arg{i}"

            argument_names.append(
                name
            )

            decode_lines.append(
                f"""
        var {name} {parameter.type}

        if err := json.Unmarshal(
            request.Args[{i}],
            &{name},
        ); err != nil {{
            writeError(
                writer,
                err,
            )
            continue
        }}
"""
            )

        decode_block = "\n".join(
            decode_lines
        )

        arguments = ", ".join(
            argument_names
        )

        runner = f"""package main

import (
    "bufio"
    "encoding/json"
    "fmt"
    "os"
)


type Request struct {{
    Args []json.RawMessage `json:"args"`
}}


type Response struct {{
    OK     bool        `json:"ok"`
    Result interface{{}} `json:"result,omitempty"`
    Error  string      `json:"error,omitempty"`
}}


func writeError(
    writer *bufio.Writer,
    err error,
) {{
    response := Response{{
        OK: false,
        Error: err.Error(),
    }}

    data, marshalErr := json.Marshal(
        response,
    )

    if marshalErr != nil {{
        fmt.Fprintln(
            writer,
            `{{"ok":false,"error":"failed to encode error response"}}`,
        )

        writer.Flush()
        return
    }}

    fmt.Fprintln(
        writer,
        string(data),
    )

    writer.Flush()
}}


func main() {{
    scanner := bufio.NewScanner(
        os.Stdin,
    )

    scanner.Buffer(
        make([]byte, 1024),
        64*1024*1024,
    )

    writer := bufio.NewWriter(
        os.Stdout,
    )

    for scanner.Scan() {{
        var request Request

        if err := json.Unmarshal(
            scanner.Bytes(),
            &request,
        ); err != nil {{
            writeError(
                writer,
                err,
            )
            continue
        }}

        if len(request.Args) != {len(self.interface.parameters)} {{
            writeError(
                writer,
                fmt.Errorf(
                    "expected {len(self.interface.parameters)} arguments, got %d",
                    len(request.Args),
                ),
            )
            continue
        }}

{decode_block}

        result := {self.interface.callable_name}({arguments})

        response := Response{{
            OK: true,
            Result: result,
        }}

        data, err := json.Marshal(
            response,
        )

        if err != nil {{
            writeError(
                writer,
                err,
            )
            continue
        }}

        fmt.Fprintln(
            writer,
            string(data),
        )

        writer.Flush()
    }}

    if err := scanner.Err(); err != nil {{
        fmt.Fprintln(
            os.Stderr,
            err,
        )
    }}
}}
"""

        runner_path.write_text(
            runner,
            encoding="utf-8",
        )

        compile_result = subprocess.run(
            [
                "go",
                "build",
                "-o",
                str(binary_path),
                str(solution_path),
                str(runner_path),
            ],
            cwd=self.workspace,
            text=True,
            capture_output=True,
        )

        if compile_result.returncode != 0:
            raise RuntimeError(
                "Failed to compile Go candidate:\n"
                + compile_result.stdout
                + "\n"
                + compile_result.stderr
            )

        self._start_process(
            [
                str(binary_path),
            ]
        )