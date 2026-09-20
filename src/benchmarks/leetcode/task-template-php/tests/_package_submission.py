#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(
    os.environ.get(
        "LEETCODE_WORKSPACE",
        "/workspace",
    )
)

LANGUAGE = "javascript"
SOURCE_FILE = "solution.js"
TARGET = WORKSPACE / "solution.json"

if TARGET.exists():
    try:
        submission = json.loads(
            TARGET.read_text()
        )

        if (
            str(submission.get("language")) == LANGUAGE
            and str(submission.get("code", "")).strip()
        ):
            print(
                "[package] existing solution.json valid; keeping"
            )
            sys.exit(0)

    except Exception:
        pass

source = WORKSPACE / SOURCE_FILE

if not source.exists():
    print(
        f"[package] source not found: {source}",
        file=sys.stderr,
    )
    sys.exit(1)

code = source.read_text().strip()

if not code:
    print(
        f"[package] source empty: {source}",
        file=sys.stderr,
    )
    sys.exit(1)

TARGET.write_text(
    json.dumps(
        {
            "language": LANGUAGE,
            "code": code,
        }
    )
)

print(
    f"[package] packaged {source.name} "
    f"-> solution.json ({len(code)} chars)"
)
