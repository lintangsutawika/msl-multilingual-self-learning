from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Parameter:
    name: str
    type: str


@dataclass(frozen=True)
class SolutionInterface:
    question_id: int
    language: str

    # Examples:
    # Python/C++/Java: "Solution"
    # Go/JavaScript/TypeScript/C/Ruby/PHP: often None
    container: str | None

    callable_name: str
    parameters: tuple[Parameter, ...]
    return_type: str

    # Where did this interface come from?
    # "doocs" or "derived"
    source: str

    # Optional raw signature for debugging/auditing
    raw_signature: str | None = None