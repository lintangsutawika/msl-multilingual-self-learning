from __future__ import annotations

from ..interfaces.models import SolutionInterface


LANGUAGE_DISPLAY_NAMES = {
    "python": "Python",
    "cpp": "C++",
    "go": "Go",
    "java": "Java",
}


def _format_parameters(
    interface: SolutionInterface,
) -> str:
    if not interface.parameters:
        return "- none"

    return "\n".join(
        (
            f"- {parameter.name}: "
            f"{parameter.type}"
        )
        for parameter in interface.parameters
    )


def _format_container(
    interface: SolutionInterface,
) -> str:
    if interface.container is None:
        return "- none"

    return f"- {interface.container}"


def render_prompt(
    problem,
    interface: SolutionInterface,
) -> str:
    language_name = LANGUAGE_DISPLAY_NAMES.get(
        interface.language
    )

    if language_name is None:
        raise ValueError(
            f"Unsupported language: "
            f"{interface.language}"
        )

    parameters = _format_parameters(
        interface
    )

    container = _format_container(
        interface
    )

    return f"""{problem.problem_description.strip()}

Required callable interface:

Container:
{container}

Callable:
- {interface.callable_name}

Parameters:
{parameters}

Return type:
- {interface.return_type}

Write a complete, self-contained, valid {language_name} solution that implements exactly this interface.
Include any imports, includes, or standard-library dependencies required by your implementation.
Return only the source code. Do not include Markdown fences or explanations.
"""