from __future__ import annotations

from .derive import derive_interface
from .models import Parameter, SolutionInterface
from .parse_readme import interface_from_doocs


SUPPORTED_LANGUAGES = (
    "python",
    "cpp",
    "go",
    "java",
)


def _merge_missing_types(
    doocs: SolutionInterface,
    derived: SolutionInterface,
) -> SolutionInterface:
    """
    Fill missing type information in a Doocs interface using the
    deterministically derived HF interface.

    Doocs remains authoritative whenever it supplies an actual type.
    The derived interface is used only for missing/unknown fields.
    """

    derived_parameters = {
        parameter.name: parameter
        for parameter in derived.parameters
    }

    merged_parameters: list[Parameter] = []

    for parameter in doocs.parameters:
        if parameter.type != "unknown":
            merged_parameters.append(
                parameter
            )
            continue

        derived_parameter = (
            derived_parameters.get(
                parameter.name
            )
        )

        if derived_parameter is None:
            merged_parameters.append(
                parameter
            )
            continue

        merged_parameters.append(
            Parameter(
                name=parameter.name,
                type=derived_parameter.type,
            )
        )

    return_type = doocs.return_type

    if return_type in {
        "",
        "unknown",
    }:
        return_type = (
            derived.return_type
        )

    return SolutionInterface(
        question_id=doocs.question_id,
        language=doocs.language,
        container=doocs.container,
        callable_name=doocs.callable_name,
        parameters=tuple(
            merged_parameters
        ),
        return_type=return_type,
        source=doocs.source,
        raw_signature=doocs.raw_signature,
    )


def resolve_interface(
    problem,
    language: str,
) -> SolutionInterface:
    """
    Resolve the callable interface for one problem/language pair.

    Resolution strategy:

    1. Derive a deterministic interface from the canonical HF Python
       starter code. This establishes the canonical callable identity
       and provides fallback type information.

    2. Find that exact callable in the corresponding Doocs language
       solution so native language-specific types can be preserved.

    3. Fill only missing Doocs type information from the derived
       interface.

    4. If the Doocs callable is unavailable or cannot be parsed,
       return the fully derived interface.
    """

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported: "
            f"{', '.join(SUPPORTED_LANGUAGES)}"
        )

    derived = derive_interface(
        problem.question_id,
        problem.starter_code,
        language,
    )

    try:
        doocs = interface_from_doocs(
            problem.question_id,
            language,
            callable_name=(
                derived.callable_name
            ),
        )

    except (
        ValueError,
        TypeError,
    ):
        doocs = None

    if doocs is None:
        return derived

    return _merge_missing_types(
        doocs,
        derived,
    )