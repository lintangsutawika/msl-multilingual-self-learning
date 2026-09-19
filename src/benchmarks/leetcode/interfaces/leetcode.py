from __future__ import annotations

import ast
import re

from .models import Parameter, SolutionInterface


# Framework language name -> LeetCode codeSnippets langSlug.
LANGUAGE_SLUGS = {
    "python": "python3",
    "cpp": "cpp",
    "rust": "rust",
    "javascript": "javascript",
    "typescript": "typescript",
    "go": "golang",
    "java": "java",
    "php": "php",
    "ruby": "ruby",
}


def _split_top_level_commas(text: str) -> list[str]:
    """Split parameters without splitting inside nested generic/container types."""
    items: list[str] = []
    current: list[str] = []

    angle = 0
    paren = 0
    bracket = 0
    brace = 0

    for char in text:
        if char == "<":
            angle += 1
        elif char == ">":
            angle = max(0, angle - 1)
        elif char == "(":
            paren += 1
        elif char == ")":
            paren = max(0, paren - 1)
        elif char == "[":
            bracket += 1
        elif char == "]":
            bracket = max(0, bracket - 1)
        elif char == "{":
            brace += 1
        elif char == "}":
            brace = max(0, brace - 1)

        if (
            char == ","
            and angle == 0
            and paren == 0
            and bracket == 0
            and brace == 0
        ):
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
        else:
            current.append(char)

    final = "".join(current).strip()
    if final:
        items.append(final)

    return items


def _parse_type_before_name(
    text: str,
) -> Parameter:
    """
    Parse forms such as:

        int n
        int[][] queries
        vector<vector<int>>& queries
        List<Integer> nums
        TreeNode* root
    """
    text = text.strip()

    match = re.match(
        r"^(?P<type>.+?)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)$",
        text,
    )

    if match is None:
        raise ValueError(
            f"Could not parse parameter: {text!r}"
        )

    return Parameter(
        name=match.group("name"),
        type=match.group("type").strip(),
    )


def parse_python_interface(
    question_id: int,
    code: str,
) -> SolutionInterface | None:
    pattern = re.compile(
        r"""
        ^class\s+Solution\s*:
        .*?
        ^[ \t]+def
        \s+
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        \s*
        \(
            (?P<params>.*?)
        \)
        \s*
        ->
        \s*
        (?P<return>[^:\n]+)
        \s*
        :
        """,
        re.VERBOSE | re.DOTALL | re.MULTILINE,
    )

    match = pattern.search(code)

    if match is None:
        return None

    parameters: list[Parameter] = []

    for item in _split_top_level_commas(
        match.group("params")
    ):
        item = item.strip()

        if item == "self":
            continue

        if "=" in item:
            item = item.split("=", 1)[0].strip()

        if ":" not in item:
            raise ValueError(
                f"Could not parse Python parameter: {item!r}"
            )

        name, type_name = item.split(":", 1)

        parameters.append(
            Parameter(
                name=name.strip(),
                type=type_name.strip(),
            )
        )

    return SolutionInterface(
        question_id=question_id,
        language="python",
        container="Solution",
        callable_name=match.group("name").strip(),
        parameters=tuple(parameters),
        return_type=match.group("return").strip(),
        source="leetcode",
        raw_signature=(
            f"def {match.group('name').strip()}"
            f"({match.group('params').strip()}) "
            f"-> {match.group('return').strip()}:"
        ),
    )

def parse_cpp_interface(
    question_id: int,
    code: str,
) -> SolutionInterface | None:
    pattern = re.compile(
        r"""
        (?:
            public:
            \s*
        )?
        (?P<return>
            [A-Za-z_]
            [A-Za-z0-9_:<>,\s*&]*
        )
        \s+
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        \s*
        \(
            (?P<params>.*?)
        \)
        \s*
        \{
        """,
        re.VERBOSE | re.DOTALL,
    )

    matches = list(
        pattern.finditer(code)
    )

    if not matches:
        return None

    # LeetCode Solution snippets normally contain one method.
    # Choose the last match so "public:" cannot become part of
    # the return type.
    match = matches[-1]

    return_type = match.group("return").strip()

    if return_type.startswith("public:"):
        return_type = return_type[len("public:"):].strip()

    parameters = tuple(
        _parse_type_before_name(item)
        for item in _split_top_level_commas(
            match.group("params")
        )
        if item.strip()
    )

    return SolutionInterface(
        question_id=question_id,
        language="cpp",
        container="Solution",
        callable_name=match.group("name").strip(),
        parameters=parameters,
        return_type=return_type,
        source="leetcode",
        raw_signature=(
            f"{return_type} "
            f"{match.group('name').strip()}"
            f"({match.group('params').strip()}) {{"
        ),
    )

def parse_rust_interface(
    question_id: int,
    code: str,
) -> SolutionInterface | None:
    pattern = re.compile(
        r"""
        pub
        \s+
        fn
        \s+
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        \s*
        \(
            (?P<params>.*?)
        \)
        \s*
        ->
        \s*
        (?P<return>[^{\n]+)
        \s*
        \{
        """,
        re.VERBOSE | re.DOTALL,
    )

    match = pattern.search(code)

    if match is None:
        return None

    parameters: list[Parameter] = []

    for item in _split_top_level_commas(
        match.group("params")
    ):
        if ":" not in item:
            raise ValueError(
                f"Could not parse Rust parameter: {item!r}"
            )

        name, type_name = item.split(":", 1)

        parameters.append(
            Parameter(
                name=name.strip(),
                type=type_name.strip(),
            )
        )

    return SolutionInterface(
        question_id=question_id,
        language="rust",
        container="Solution",
        callable_name=match.group("name").strip(),
        parameters=tuple(parameters),
        return_type=match.group("return").strip(),
        source="leetcode",
        raw_signature=match.group(0).strip(),
    )


def parse_typescript_interface(
    question_id: int,
    code: str,
) -> SolutionInterface | None:
    pattern = re.compile(
        r"""
        function
        \s+
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        \s*
        \(
            (?P<params>.*?)
        \)
        \s*
        :
        \s*
        (?P<return>[^{\n]+)
        \s*
        \{
        """,
        re.VERBOSE | re.DOTALL,
    )

    match = pattern.search(code)

    if match is None:
        return None

    parameters: list[Parameter] = []

    for item in _split_top_level_commas(
        match.group("params")
    ):
        if ":" not in item:
            raise ValueError(
                f"Could not parse TypeScript parameter: {item!r}"
            )

        name, type_name = item.split(":", 1)

        parameters.append(
            Parameter(
                name=name.strip(),
                type=type_name.strip(),
            )
        )

    return SolutionInterface(
        question_id=question_id,
        language="typescript",
        container=None,
        callable_name=match.group("name").strip(),
        parameters=tuple(parameters),
        return_type=match.group("return").strip(),
        source="leetcode",
        raw_signature=match.group(0).strip(),
    )


def parse_go_interface(
    question_id: int,
    code: str,
) -> SolutionInterface | None:
    pattern = re.compile(
        r"""
        func
        \s+
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        \s*
        \(
            (?P<params>.*?)
        \)
        \s*
        (?P<return>[^{\n]+)
        \s*
        \{
        """,
        re.VERBOSE | re.DOTALL,
    )

    match = pattern.search(code)

    if match is None:
        return None

    parameters: list[Parameter] = []

    for item in _split_top_level_commas(
        match.group("params")
    ):
        item = item.strip()

        parameter_match = re.match(
            r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+(?P<type>.+)$",
            item,
        )

        if parameter_match is None:
            raise ValueError(
                f"Could not parse Go parameter: {item!r}"
            )

        parameters.append(
            Parameter(
                name=parameter_match.group("name"),
                type=parameter_match.group("type").strip(),
            )
        )

    return SolutionInterface(
        question_id=question_id,
        language="go",
        container=None,
        callable_name=match.group("name").strip(),
        parameters=tuple(parameters),
        return_type=match.group("return").strip(),
        source="leetcode",
        raw_signature=match.group(0).strip(),
    )


def parse_java_interface(
    question_id: int,
    code: str,
) -> SolutionInterface | None:
    pattern = re.compile(
        r"""
        public
        \s+
        (?P<return>.+?)
        \s+
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        \s*
        \(
            (?P<params>.*?)
        \)
        \s*
        \{
        """,
        re.VERBOSE | re.DOTALL,
    )

    match = pattern.search(code)

    if match is None:
        return None

    parameters = tuple(
        _parse_type_before_name(item)
        for item in _split_top_level_commas(
            match.group("params")
        )
        if item.strip()
    )

    return SolutionInterface(
        question_id=question_id,
        language="java",
        container="Solution",
        callable_name=match.group("name").strip(),
        parameters=parameters,
        return_type=match.group("return").strip(),
        source="leetcode",
        raw_signature=match.group(0).strip(),
    )


def parse_javascript_interface(
    question_id: int,
    code: str,
) -> SolutionInterface | None:
    function_match = re.search(
        r"""
        var
        \s+
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        \s*=\s*
        function
        \s*
        \(
            (?P<params>.*?)
        \)
        \s*
        \{
        """,
        code,
        re.VERBOSE | re.DOTALL,
    )

    if function_match is None:
        return None

    comment_parameters = {
        name: type_name
        for type_name, name in re.findall(
            r"@param\s+\{([^}]+)\}\s+([A-Za-z_][A-Za-z0-9_]*)",
            code,
        )
    }

    return_match = re.search(
        r"@return\s+\{([^}]+)\}",
        code,
    )

    if return_match is None:
        return None

    names = [
        item.strip()
        for item in function_match.group("params").split(",")
        if item.strip()
    ]

    parameters = tuple(
        Parameter(
            name=name,
            type=comment_parameters[name],
        )
        for name in names
    )

    return SolutionInterface(
        question_id=question_id,
        language="javascript",
        container=None,
        callable_name=function_match.group("name").strip(),
        parameters=parameters,
        return_type=return_match.group(1).strip(),
        source="leetcode",
        raw_signature=function_match.group(0).strip(),
    )


def parse_php_interface(
    question_id: int,
    code: str,
) -> SolutionInterface | None:
    function_match = re.search(
        r"""
        ^[ \t]*function
        \s+
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        \s*
        \(
            (?P<params>.*?)
        \)
        \s*
        \{
        """,
        code,
        re.VERBOSE | re.DOTALL | re.MULTILINE,
    )

    if function_match is None:
        return None

    comment_parameters = {
        name: type_name
        for type_name, name in re.findall(
            r"@param\s+([^\s]+)\s+\$([A-Za-z_][A-Za-z0-9_]*)",
            code,
        )
    }

    return_match = re.search(
        r"@return\s+([^\s*]+)",
        code,
    )

    if return_match is None:
        return None

    names = [
        item.split("=", 1)[0].strip().lstrip("$")
        for item in function_match.group("params").split(",")
        if item.strip()
    ]

    parameters = tuple(
        Parameter(
            name=name,
            type=comment_parameters[name],
        )
        for name in names
    )

    return SolutionInterface(
        question_id=question_id,
        language="php",
        container="Solution",
        callable_name=function_match.group("name").strip(),
        parameters=parameters,
        return_type=return_match.group(1).strip(),
        source="leetcode",
        raw_signature=function_match.group(0).strip(),
    )

def parse_ruby_interface(
    question_id: int,
    code: str,
) -> SolutionInterface | None:
    function_match = re.search(
        r"""
        ^def
        \s+
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        \s*
        \(
            (?P<params>.*?)
        \)
        """,
        code,
        re.VERBOSE | re.DOTALL | re.MULTILINE,
    )

    if function_match is None:
        return None

    comment_parameters = {
        name: type_name
        for type_name, name in re.findall(
            r"#\s*@param\s+\{([^}]+)\}\s+([A-Za-z_][A-Za-z0-9_]*)",
            code,
        )
    }

    return_match = re.search(
        r"#\s*@return\s+\{([^}]+)\}",
        code,
    )

    if return_match is None:
        return None

    names = [
        item.split("=", 1)[0].strip()
        for item in function_match.group("params").split(",")
        if item.strip()
    ]

    parameters = tuple(
        Parameter(
            name=name,
            type=comment_parameters[name],
        )
        for name in names
    )

    return SolutionInterface(
        question_id=question_id,
        language="ruby",
        container=None,
        callable_name=function_match.group("name").strip(),
        parameters=parameters,
        return_type=return_match.group(1).strip(),
        source="leetcode",
        raw_signature=function_match.group(0).strip(),
    )

PARSERS = {
    "python": parse_python_interface,
    "cpp": parse_cpp_interface,
    "rust": parse_rust_interface,
    "javascript": parse_javascript_interface,
    "typescript": parse_typescript_interface,
    "go": parse_go_interface,
    "java": parse_java_interface,
    "php": parse_php_interface,
    "ruby": parse_ruby_interface,
}


def parse_leetcode_interface(
    question_id: int,
    language: str,
    code: str,
) -> SolutionInterface | None:
    try:
        parser = PARSERS[language]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported language: {language}"
        ) from exc

    return parser(
        question_id,
        code,
    )
