from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import load_test_split
from .python.generate_task import generate_python_task


def generate_python_dataset(
    output_root: Path,
    limit: int | None = None,
) -> list[Path]:
    problems = load_test_split()

    if limit is not None:
        problems = problems[:limit]

    generated = []

    for problem in problems:
        task_dir = generate_python_task(
            problem,
            output_root,
        )

        generated.append(task_dir)

        print(f"generated: {problem.task_id}")

    return generated


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where Harbor tasks will be generated.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Generate only the first N problems.",
    )

    args = parser.parse_args()

    generated = generate_python_dataset(
        output_root=args.output,
        limit=args.limit,
    )

    print()
    print(f"Generated {len(generated)} tasks.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()