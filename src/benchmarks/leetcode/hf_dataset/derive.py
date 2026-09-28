from __future__ import annotations

import ast
from dataclasses import dataclass

from .models import Parameter, SolutionInterface


# ---------------------------------------------------------------------------
# Canonical interface parsed from HF Python starter code
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalParameter:
    name: str
    type: str


@dataclass(frozen=True)
class CanonicalSignature:
    container: str | None
    callable_name: str
    parameters: tuple[CanonicalParameter, ...]
    return_type: str


# ---------------------------------------------------------------------------
# Python annotation helpers
# ---------------------------------------------------------------------------


def annotation_to_string(
    node: ast.expr | None,
) -> str:
    if node is None:
        raise ValueError(
            "Missing type annotation in HF starter code"
        )

    return ast.unparse(node)


def normalize_type(
    type_name: str,
) -> str:
    """
    Normalize Python annotations into a small canonical representation.

    Examples:
        typing.List[int]       -> List[int]
        List [ int ]           -> List[int]
        Optional [ TreeNode ]  -> Optional[TreeNode]
    """
    return (
        type_name
        .replace("typing.", "")
        .replace(" ", "")
    )


# ---------------------------------------------------------------------------
# HF starter-code repair
# ---------------------------------------------------------------------------


def _make_starter_code_parseable(
    starter_code: str,
) -> str:
    """
    Make HF starter snippets syntactically valid Python.

    Some LeetCode starter-code snippets contain only a function
    declaration with no body:

        class Solution:
            def foo(self, x: int) -> int:

    Python's AST parser rejects this. Since we only need the signature,
    we deterministically insert `pass` when a function body is missing.
    """
    lines = starter_code.splitlines()
    fixed: list[str] = []

    for index, line in enumerate(lines):
        fixed.append(line)

        stripped = line.strip()

        if not stripped.startswith(
            ("def ", "async def ")
        ):
            continue

        if not stripped.endswith(":"):
            continue

        current_indent = (
            len(line)
            - len(line.lstrip())
        )

        next_nonempty: str | None = None

        for later_line in lines[index + 1:]:
            if later_line.strip():
                next_nonempty = later_line
                break

        needs_body = False

        if next_nonempty is None:
            needs_body = True

        else:
            next_indent = (
                len(next_nonempty)
                - len(next_nonempty.lstrip())
            )

            if next_indent <= current_indent:
                needs_body = True

        if needs_body:
            fixed.append(
                " " * (current_indent + 4)
                + "pass"
            )

    return "\n".join(fixed)


# ---------------------------------------------------------------------------
# Parse HF Python signature
# ---------------------------------------------------------------------------


def _parameters_from_function(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    skip_self: bool,
) -> tuple[CanonicalParameter, ...]:
    parameters: list[CanonicalParameter] = []

    for arg in function.args.args:
        if skip_self and arg.arg == "self":
            continue

        parameters.append(
            CanonicalParameter(
                name=arg.arg,
                type=normalize_type(
                    annotation_to_string(
                        arg.annotation
                    )
                ),
            )
        )

    return tuple(parameters)


def parse_hf_python_signature(
    starter_code: str,
) -> CanonicalSignature:
    """
    Recover the callable interface from HF LeetCode starter_code.

    The starter code is used only internally for benchmark metadata.
    It is never shown to the evaluated model.
    """
    try:
        tree = ast.parse(
            starter_code
        )

    except SyntaxError:
        repaired = _make_starter_code_parseable(
            starter_code
        )

        tree = ast.parse(
            repaired
        )

    # Prefer the standard LeetCode structure:
    #
    # class Solution:
    #     def method(...):
    #
    for node in tree.body:
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "Solution"
        ):
            for child in node.body:
                if isinstance(
                    child,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                ):
                    return CanonicalSignature(
                        container="Solution",
                        callable_name=child.name,
                        parameters=_parameters_from_function(
                            child,
                            skip_self=True,
                        ),
                        return_type=normalize_type(
                            annotation_to_string(
                                child.returns
                            )
                        ),
                    )

    # Handle the rare case where HF gives a top-level function.
    for node in tree.body:
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            return CanonicalSignature(
                container=None,
                callable_name=node.name,
                parameters=_parameters_from_function(
                    node,
                    skip_self=False,
                ),
                return_type=normalize_type(
                    annotation_to_string(
                        node.returns
                    )
                ),
            )

    raise ValueError(
        "Could not find callable interface "
        "in HF starter_code"
    )


# ---------------------------------------------------------------------------
# Deterministic type mappings
# ---------------------------------------------------------------------------


PYTHON_TYPE_MAP = {
    "int": "int",
    "str": "str",
    "bool": "bool",
    "float": "float",

    "List[int]": "List[int]",
    "List[str]": "List[str]",
    "List[bool]": "List[bool]",
    "List[float]": "List[float]",

    "List[List[int]]": "List[List[int]]",
    "List[List[str]]": "List[List[str]]",

    "Optional[TreeNode]": "Optional[TreeNode]",
}


CPP_TYPE_MAP = {
    "int": "int",
    "str": "string",
    "bool": "bool",
    "float": "double",

    "List[int]": "vector<int>",
    "List[str]": "vector<string>",
    "List[bool]": "vector<bool>",
    "List[float]": "vector<double>",

    "List[List[int]]": "vector<vector<int>>",
    "List[List[str]]": "vector<vector<string>>",

    "Optional[TreeNode]": "TreeNode*",
}


GO_TYPE_MAP = {
    "int": "int",
    "str": "string",
    "bool": "bool",
    "float": "float64",

    "List[int]": "[]int",
    "List[str]": "[]string",
    "List[bool]": "[]bool",
    "List[float]": "[]float64",

    "List[List[int]]": "[][]int",
    "List[List[str]]": "[][]string",

    "Optional[TreeNode]": "*TreeNode",
}


JAVA_TYPE_MAP = {
    "int": "int",
    "str": "String",
    "bool": "boolean",
    "float": "double",

    "List[int]": "int[]",
    "List[str]": "String[]",
    "List[bool]": "boolean[]",
    "List[float]": "double[]",

    "List[List[int]]": "int[][]",
    "List[List[str]]": "String[][]",

    "Optional[TreeNode]": "TreeNode",
}


TYPE_MAPS = {
    "python": PYTHON_TYPE_MAP,
    "cpp": CPP_TYPE_MAP,
    "go": GO_TYPE_MAP,
    "java": JAVA_TYPE_MAP,
}


# ---------------------------------------------------------------------------
# Type conversion
# ---------------------------------------------------------------------------


def map_type(
    type_name: str,
    language: str,
) -> str:
    if language not in TYPE_MAPS:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported: "
            f"{', '.join(sorted(TYPE_MAPS))}"
        )

    mapping = TYPE_MAPS[language]

    if type_name not in mapping:
        raise ValueError(
            f"Unsupported HF type "
            f"{type_name!r} "
            f"for language {language!r}"
        )

    return mapping[type_name]


# ---------------------------------------------------------------------------
# Language-specific structural conventions
# ---------------------------------------------------------------------------


def get_container(
    language: str,
) -> str | None:
    """
    Return the standard LeetCode callable container for each language.

    Python:
        class Solution

    C++:
        class Solution

    Java:
        class Solution

    Go:
        top-level function
    """
    if language in {
        "python",
        "cpp",
        "java",
    }:
        return "Solution"

    if language == "go":
        return None

    raise ValueError(
        f"Unsupported language: {language}"
    )


# ---------------------------------------------------------------------------
# Public derivation function
# ---------------------------------------------------------------------------


def derive_interface(
    question_id: int,
    starter_code: str,
    language: str,
) -> SolutionInterface:
    """
    Deterministically derive a language-specific callable interface
    from the HF Python starter-code signature.

    No model is involved.

    This is used only when the Doocs README does not provide a usable
    interface.
    """
    if language not in TYPE_MAPS:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported: "
            f"{', '.join(sorted(TYPE_MAPS))}"
        )

    signature = parse_hf_python_signature(
        starter_code
    )

    parameters = tuple(
        Parameter(
            name=parameter.name,
            type=map_type(
                parameter.type,
                language,
            ),
        )
        for parameter in signature.parameters
    )

    return_type = map_type(
        signature.return_type,
        language,
    )

    return SolutionInterface(
        question_id=question_id,
        language=language,
        container=get_container(
            language
        ),
        callable_name=signature.callable_name,
        parameters=parameters,
        return_type=return_type,
        source="derived",
        raw_signature=None,
    )