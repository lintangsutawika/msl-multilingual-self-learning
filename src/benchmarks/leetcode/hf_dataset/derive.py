"""Parameter names and types from newfacade/LeetCodeDataset's Python starter code.

The canonical tests call candidate(name=value, ...) with these names, so the
build records them (metadata.canonical_parameter_names) for every language.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

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
