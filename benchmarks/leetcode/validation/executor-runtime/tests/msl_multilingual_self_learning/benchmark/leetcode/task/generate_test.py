from __future__ import annotations

import textwrap

from msl_multilingual_self_learning.benchmark.leetcode.dataset import (
    LeetCodeProblem,
)
from msl_multilingual_self_learning.benchmark.leetcode.interfaces.models import (
    SolutionInterface,
)


def _render_interface_literal(
    interface: SolutionInterface,
) -> str:
    parameters = ",\n".join(
        (
            "        Parameter("
            f"name={parameter.name!r}, "
            f"type={parameter.type!r}"
            ")"
        )
        for parameter in interface.parameters
    )

    if parameters:
        parameters_block = (
            "(\n"
            f"{parameters},\n"
            "    )"
        )
    else:
        parameters_block = "()"

    return f"""SolutionInterface(
    question_id={interface.question_id!r},
    language={interface.language!r},
    container={interface.container!r},
    callable_name={interface.callable_name!r},
    parameters={parameters_block},
    return_type={interface.return_type!r},
    source={interface.source!r},
    raw_signature={interface.raw_signature!r},
)"""


def render_test_py(
    problem: LeetCodeProblem,
    interface: SolutionInterface,
) -> str:
    interface_literal = _render_interface_literal(
        interface
    )

    hf_test = problem.test.rstrip()

    return f"""from __future__ import annotations

import sys
import traceback
from pathlib import Path

from msl_multilingual_self_learning.benchmark.leetcode.interfaces.models import (
    Parameter,
    SolutionInterface,
)
from msl_multilingual_self_learning.benchmark.leetcode.verifier.verifier import (
    SolutionVerifier,
)


INTERFACE = {interface_literal}


{hf_test}


def main() -> int:
    verifier = SolutionVerifier(
        workspace=Path("/workspace"),
        interface=INTERFACE,
    )

    try:
        verifier.prepare()

        def candidate(*args, **kwargs):
            if args and kwargs:
                raise TypeError(
                    "candidate calls cannot mix positional "
                    "and keyword arguments"
                )

            if kwargs:
                ordered_args = []

                for parameter in INTERFACE.parameters:
                    if parameter.name not in kwargs:
                        raise TypeError(
                            f"Missing argument: {{parameter.name}}"
                        )

                    ordered_args.append(
                        kwargs[parameter.name]
                    )

                unexpected = (
                    set(kwargs)
                    - {{
                        parameter.name
                        for parameter
                        in INTERFACE.parameters
                    }}
                )

                if unexpected:
                    raise TypeError(
                        f"Unexpected arguments: "
                        f"{{sorted(unexpected)}}"
                    )

                return verifier.call(
                    *ordered_args
                )

            return verifier.call(
                *args
            )

        try:
            check(candidate)

        except AssertionError:
            traceback.print_exc()
            return 1

        except Exception:
            traceback.print_exc()
            return 2

        return 0

    except Exception:
        traceback.print_exc()
        return 2

    finally:
        verifier.close()


if __name__ == "__main__":
    sys.exit(main())
"""