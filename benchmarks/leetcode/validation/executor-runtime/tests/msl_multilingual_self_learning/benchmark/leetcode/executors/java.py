from __future__ import annotations

import subprocess
from pathlib import Path

from .native import NativeProcessExecutor


GSON_JAR_CANDIDATES = [
    "/usr/share/java/gson.jar",
    "/usr/share/java/gson-2.10.jar",
    "/usr/share/java/gson-2.10.1.jar",
]


def _find_gson_jar() -> str:
    for candidate in GSON_JAR_CANDIDATES:
        if Path(candidate).exists():
            return candidate

    java_dir = Path(
        "/usr/share/java"
    )

    if java_dir.exists():
        matches = sorted(
            java_dir.glob(
                "gson*.jar"
            )
        )

        if matches:
            return str(
                matches[0]
            )

    raise RuntimeError(
        "Could not find Gson jar "
        "under /usr/share/java"
    )


def _java_decode_expression(
    type_name: str,
    index: int,
) -> str:
    type_name = type_name.strip()

    if "<" in type_name:
        return (
            "gson.fromJson("
            f"args.get({index}), "
            f"new TypeToken<{type_name}>() {{}}"
            ".getType())"
        )

    return (
        "gson.fromJson("
        f"args.get({index}), "
        f"{type_name}.class)"
    )


class JavaExecutor(
    NativeProcessExecutor,
):
    def prepare(
        self,
        code: str,
    ) -> None:
        solution_path = (
            self.workspace
            / "Solution.java"
        )

        runner_path = (
            self.workspace
            / "Runner.java"
        )

        solution_path.write_text(
            code,
            encoding="utf-8",
        )

        argument_lines: list[str] = []
        argument_names: list[str] = []

        for i, parameter in enumerate(
            self.interface.parameters
        ):
            name = f"arg{i}"

            argument_names.append(
                name
            )

            expression = (
                _java_decode_expression(
                    parameter.type,
                    i,
                )
            )

            argument_lines.append(
                f"""
                {parameter.type} {name} =
                    {expression};
"""
            )

        argument_block = "\n".join(
            argument_lines
        )

        arguments = ", ".join(
            argument_names
        )

        container = (
            self.interface.container
            or "Solution"
        )

        runner = f"""
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

import com.google.gson.*;
import com.google.gson.reflect.TypeToken;


public class Runner {{

    public static void main(
        String[] argsFromShell
    ) throws Exception {{

        Gson gson =
            new Gson();

        BufferedReader reader =
            new BufferedReader(
                new InputStreamReader(
                    System.in,
                    StandardCharsets.UTF_8
                )
            );

        BufferedWriter writer =
            new BufferedWriter(
                new OutputStreamWriter(
                    System.out,
                    StandardCharsets.UTF_8
                )
            );

        {container} solution =
            new {container}();

        String line;

        while (
            (line = reader.readLine())
            != null
        ) {{
            try {{
                JsonObject request =
                    JsonParser
                        .parseString(
                            line
                        )
                        .getAsJsonObject();

                JsonArray args =
                    request
                        .getAsJsonArray(
                            "args"
                        );

{argument_block}

                var result =
                    solution
                        .{self.interface.callable_name}(
                            {arguments}
                        );

                JsonObject response =
                    new JsonObject();

                response.addProperty(
                    "ok",
                    true
                );

                response.add(
                    "result",
                    gson.toJsonTree(
                        result
                    )
                );

                writer.write(
                    gson.toJson(
                        response
                    )
                );

                writer.newLine();
                writer.flush();
            }}
            catch (Throwable exc) {{
                JsonObject response =
                    new JsonObject();

                response.addProperty(
                    "ok",
                    false
                );

                response.addProperty(
                    "error",
                    exc.toString()
                );

                writer.write(
                    gson.toJson(
                        response
                    )
                );

                writer.newLine();
                writer.flush();
            }}
        }}
    }}
}}
"""

        runner_path.write_text(
            runner,
            encoding="utf-8",
        )

        classpath = (
            _find_gson_jar()
        )

        compile_result = subprocess.run(
            [
                "javac",
                "-cp",
                classpath,
                str(solution_path),
                str(runner_path),
            ],
            cwd=self.workspace,
            text=True,
            capture_output=True,
        )

        if compile_result.returncode != 0:
            raise RuntimeError(
                "Failed to compile Java candidate:\n"
                + compile_result.stdout
                + "\n"
                + compile_result.stderr
            )

        self._start_process(
            [
                "java",
                "-cp",
                (
                    str(self.workspace)
                    + ":"
                    + classpath
                ),
                "Runner",
            ]
        )