from __future__ import annotations

import re

from .doocs import find_readme
from .models import Parameter, SolutionInterface


LANGUAGE_HEADINGS = {
    "python": {"Python", "Python3"},
    "cpp": {"C++"},
    "go": {"Go"},
    "java": {"Java"},
}


HEADING_RE = re.compile(
    r"^####\s+(.+?)\s*$",
    re.MULTILINE,
)

CODE_BLOCK_RE = re.compile(
    r"```[A-Za-z0-9_+#.\-]*\s*\n(.*?)```",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# README extraction
# ---------------------------------------------------------------------------


def extract_language_section(
    readme_text: str,
    language: str,
) -> str | None:
    if language not in LANGUAGE_HEADINGS:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported: {', '.join(sorted(LANGUAGE_HEADINGS))}"
        )

    aliases = LANGUAGE_HEADINGS[language]
    headings = list(
        HEADING_RE.finditer(readme_text)
    )

    for index, match in enumerate(headings):
        heading = match.group(1).strip()

        if heading not in aliases:
            continue

        start = match.end()

        if index + 1 < len(headings):
            end = headings[index + 1].start()
        else:
            end = len(readme_text)

        return readme_text[start:end].strip()

    return None


def extract_first_code_block(
    section: str,
) -> str | None:
    match = CODE_BLOCK_RE.search(section)

    if match is None:
        return None

    return match.group(1).strip()


def _split_top_level_commas(
    text: str,
) -> list[str]:
    """
    Split parameter lists without breaking nested generic/container types.
    """

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
            continue

        current.append(char)

    final = "".join(current).strip()

    if final:
        items.append(final)

    return items


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


PYTHON_METHOD_RE = re.compile(
    r"""
    def
    \s+
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)
    \s*
    \(
        (?P<params>[^)]*)
    \)
    \s*
    (?:
        ->
        \s*
        (?P<return>[^:]+)
    )?
    \s*
    :
    """,
    re.VERBOSE,
)


def parse_python_parameters(
    text: str,
) -> tuple[Parameter, ...]:
    text = text.strip()

    if not text:
        return ()

    parameters: list[Parameter] = []

    for item in _split_top_level_commas(text):
        item = item.strip()

        if item == "self":
            continue

        if ":" in item:
            name, type_name = item.split(":", 1)

            if "=" in type_name:
                type_name = type_name.split("=", 1)[0]

            parameters.append(
                Parameter(
                    name=name.strip(),
                    type=type_name.strip(),
                )
            )
        else:
            parameters.append(
                Parameter(
                    name=item,
                    type="unknown",
                )
            )

    return tuple(parameters)


def parse_python_interface(
    question_id: int,
    code: str,
    callable_name: str,
) -> SolutionInterface | None:
    for match in PYTHON_METHOD_RE.finditer(code):
        if match.group("name").strip() != callable_name:
            continue

        return SolutionInterface(
            question_id=question_id,
            language="python",
            container="Solution",
            callable_name=match.group("name").strip(),
            parameters=parse_python_parameters(
                match.group("params")
            ),
            return_type=(
                match.group("return") or "unknown"
            ).strip(),
            source="doocs",
            raw_signature=match.group(0).strip(),
        )

    return None


# ---------------------------------------------------------------------------
# C++
# ---------------------------------------------------------------------------


CPP_METHOD_RE = re.compile(
    r"""
    ^[ \t]*
    (?P<return>
        [A-Za-z_][A-Za-z0-9_:<>\[\],*& ]*
    )
    [ \t]+
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)
    [ \t]*
    \(
        (?P<params>[^)]*)
    \)
    [ \t]*
    (?:
        const
        [ \t]*
    )?
    \{
    """,
    re.VERBOSE | re.MULTILINE,
)


def parse_cpp_parameters(
    text: str,
) -> tuple[Parameter, ...]:
    text = text.strip()

    if not text or text == "void":
        return ()

    parameters: list[Parameter] = []

    for item in _split_top_level_commas(text):
        item = item.strip()

        if "=" in item:
            item = item.split("=", 1)[0].strip()

        match = re.match(
            r"^(?P<type>.+?)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)$",
            item,
        )

        if match is None:
            match = re.match(
                r"^(?P<type>.+?[*&])(?P<name>[A-Za-z_][A-Za-z0-9_]*)$",
                item,
            )

        if match is None:
            raise ValueError(
                f"Could not parse C++ parameter: {item!r}"
            )

        parameters.append(
            Parameter(
                name=match.group("name").strip(),
                type=match.group("type").strip(),
            )
        )

    return tuple(parameters)


def parse_cpp_interface(
    question_id: int,
    code: str,
    callable_name: str,
) -> SolutionInterface | None:
    for match in CPP_METHOD_RE.finditer(code):
        if match.group("name").strip() != callable_name:
            continue

        return SolutionInterface(
            question_id=question_id,
            language="cpp",
            container="Solution",
            callable_name=match.group("name").strip(),
            parameters=parse_cpp_parameters(
                match.group("params")
            ),
            return_type=match.group("return").strip(),
            source="doocs",
            raw_signature=match.group(0).strip(),
        )

    return None


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------


GO_FUNCTION_RE = re.compile(
    r"""
    func
    \s+
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)
    \s*
    \(
        (?P<params>[^)]*)
    \)
    \s*
    (?P<return>[^{\n]+)?
    \s*
    \{
    """,
    re.VERBOSE,
)


def _normalize_go_return_type(
    return_type: str,
) -> str:
    """
    Normalize named Go return values.

    Examples:
        (ans int)      -> int
        (ans []int)    -> []int
        (ans int64)    -> int64
        (ans float64)  -> float64
    """

    value = return_type.strip()

    if (
        value.startswith("(")
        and value.endswith(")")
    ):
        inner = value[1:-1].strip()

        parts = inner.split(
            None,
            1,
        )

        if len(parts) == 2:
            _, type_name = parts
            return type_name.strip()

    return value


def parse_go_parameters(
    text: str,
) -> tuple[Parameter, ...]:
    text = text.strip()

    if not text:
        return ()

    parameters: list[Parameter] = []

    for item in _split_top_level_commas(text):
        parts = item.strip().split()

        if len(parts) < 2:
            raise ValueError(
                f"Could not parse Go parameter: {item!r}"
            )

        name = parts[0]
        type_name = " ".join(parts[1:])

        parameters.append(
            Parameter(
                name=name.strip(),
                type=type_name.strip(),
            )
        )

    return tuple(parameters)


def parse_go_interface(
    question_id: int,
    code: str,
    callable_name: str,
) -> SolutionInterface | None:
    for match in GO_FUNCTION_RE.finditer(code):
        if match.group("name").strip() != callable_name:
            continue

        raw_return_type = (
            match.group("return") or ""
        ).strip()

        return SolutionInterface(
            question_id=question_id,
            language="go",
            container=None,
            callable_name=match.group("name").strip(),
            parameters=parse_go_parameters(
                match.group("params")
            ),
            return_type=_normalize_go_return_type(
                raw_return_type
            ),
            source="doocs",
            raw_signature=match.group(0).strip(),
        )

    return None


# ---------------------------------------------------------------------------
# Java
# ---------------------------------------------------------------------------


JAVA_METHOD_RE = re.compile(
    r"""
    (?:
        public
        \s+
    )?
    (?:
        static
        \s+
    )?
    (?P<return>
        [A-Za-z_][A-Za-z0-9_<>\[\].?]*
    )
    \s+
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)
    \s*
    \(
        (?P<params>[^)]*)
    \)
    \s*
    \{
    """,
    re.VERBOSE,
)


def parse_java_parameters(
    text: str,
) -> tuple[Parameter, ...]:
    text = text.strip()

    if not text:
        return ()

    parameters: list[Parameter] = []

    for item in _split_top_level_commas(text):
        parts = item.strip().rsplit(
            maxsplit=1
        )

        if len(parts) != 2:
            raise ValueError(
                f"Could not parse Java parameter: {item!r}"
            )

        type_name, name = parts

        parameters.append(
            Parameter(
                name=name.strip(),
                type=type_name.strip(),
            )
        )

    return tuple(parameters)


def parse_java_interface(
    question_id: int,
    code: str,
    callable_name: str,
) -> SolutionInterface | None:
    for match in JAVA_METHOD_RE.finditer(code):
        if match.group("name").strip() != callable_name:
            continue

        return SolutionInterface(
            question_id=question_id,
            language="java",
            container="Solution",
            callable_name=match.group("name").strip(),
            parameters=parse_java_parameters(
                match.group("params")
            ),
            return_type=match.group("return").strip(),
            source="doocs",
            raw_signature=match.group(0).strip(),
        )

    return None


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


PARSERS = {
    "python": parse_python_interface,
    "cpp": parse_cpp_interface,
    "go": parse_go_interface,
    "java": parse_java_interface,
}


def interface_from_doocs(
    question_id: int,
    language: str,
    callable_name: str,
) -> SolutionInterface | None:
    """
    Resolve a language-specific callable interface from Doocs README_EN.md.

    The parser searches specifically for callable_name instead of taking
    the first function or method in the language code block.

    If the README, language section, code block, or requested callable
    cannot be found, this function returns None.
    """

    if language not in PARSERS:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported: {', '.join(sorted(PARSERS))}"
        )

    readme = find_readme(
        question_id
    )

    if readme is None:
        return None

    readme_text = readme.read_text(
        encoding="utf-8",
        errors="replace",
    )

    section = extract_language_section(
        readme_text,
        language,
    )

    if section is None:
        return None

    code = extract_first_code_block(
        section
    )

    if code is None:
        return None

    parser = PARSERS[
        language
    ]

    return parser(
        question_id,
        code,
        callable_name,
    )