from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .native import NativeProcessExecutor


def _cpp_value_type(
    type_name: str,
) -> str:
    value = type_name.strip()

    value = re.sub(
        r"\bconst\b",
        "",
        value,
    )

    value = value.replace(
        "&",
        "",
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


def _cpp_decode_statement(
    index: int,
    name: str,
    type_name: str,
) -> str:
    value_type = _cpp_value_type(
        type_name
    )

    if "TreeNode" in value_type:
        return (
            f"TreeNode* {name} = "
            f"build_tree(args.at({index}));"
        )

    return (
        f"{value_type} {name} = "
        f"args.at({index}).get<{value_type}>();"
    )


class CppExecutor(
    NativeProcessExecutor,
):
    def prepare(
        self,
        code: str,
    ) -> None:
        solution_path = (
            self.workspace
            / "solution.cpp"
        )

        runner_path = (
            self.workspace
            / "runner.cpp"
        )

        binary_path = (
            self.workspace
            / "runner"
        )

        solution_path.write_text(
            code,
            encoding="utf-8",
        )

        parameter_names = [
            f"arg{i}"
            for i, _ in enumerate(
                self.interface.parameters
            )
        ]

        decode_lines = []

        for i, parameter in enumerate(
            self.interface.parameters
        ):
            decode_lines.append(
                _cpp_decode_statement(
                    index=i,
                    name=parameter_names[i],
                    type_name=parameter.type,
                )
            )

        arguments = ", ".join(
            parameter_names
        )

        has_tree = any(
            "TreeNode" in parameter.type
            for parameter
            in self.interface.parameters
        )

        tree_helper = ""

        if has_tree:
            tree_helper = r"""
TreeNode* build_tree(const json& value) {
    if (value.is_null()) {
        return nullptr;
    }

    TreeNode* node = new TreeNode(
        value.at("val").get<int>()
    );

    if (value.contains("left")) {
        node->left = build_tree(
            value.at("left")
        );
    }

    if (value.contains("right")) {
        node->right = build_tree(
            value.at("right")
        );
    }

    return node;
}
"""

        decode_block = "\n            ".join(
            decode_lines
        )

        runner = f"""
#include <bits/stdc++.h>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

#include "solution.cpp"

{tree_helper}

int main() {{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Solution solution;

    string line;

    while (getline(cin, line)) {{
        try {{
            json request = json::parse(line);
            const json& args = request.at("args");

            {decode_block}

            auto result = solution.{self.interface.callable_name}(
                {arguments}
            );

            json response = {{
                {{"ok", true}},
                {{"result", result}}
            }};

            cout
                << response.dump()
                << '\\n'
                << flush;
        }}
        catch (const exception& exc) {{
            json response = {{
                {{"ok", false}},
                {{"error", exc.what()}}
            }};

            cout
                << response.dump()
                << '\\n'
                << flush;
        }}
        catch (...) {{
            json response = {{
                {{"ok", false}},
                {{"error", "unknown C++ exception"}}
            }};

            cout
                << response.dump()
                << '\\n'
                << flush;
        }}
    }}

    return 0;
}}
"""

        runner_path.write_text(
            runner,
            encoding="utf-8",
        )

        compile_result = subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-O2",
                "-pipe",
                str(runner_path),
                "-o",
                str(binary_path),
            ],
            cwd=self.workspace,
            text=True,
            capture_output=True,
        )

        if compile_result.returncode != 0:
            raise RuntimeError(
                "Failed to compile C++ candidate:\n"
                + compile_result.stdout
                + "\n"
                + compile_result.stderr
            )

        self._start_process(
            [str(binary_path)]
        )