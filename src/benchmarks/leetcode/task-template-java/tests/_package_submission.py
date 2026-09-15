#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("LEETCODE_WORKSPACE", "/workspace"))
LANGUAGE = "java"
SOURCE_FILE = "Solution.java"
TARGET = WORKSPACE / "solution.json"

if TARGET.exists():
    try:
        sub = json.loads(TARGET.read_text())
        if str(sub.get("language")) == LANGUAGE and str(sub.get("code", "")).strip():
            print("[package] existing solution.json valid; keeping")
            sys.exit(0)
    except Exception:
        pass

src = WORKSPACE / SOURCE_FILE
if not src.exists():
    print(f"[package] source not found: {src}", file=sys.stderr)
    sys.exit(1)
code = src.read_text().strip()
if not code:
    print(f"[package] source empty: {src}", file=sys.stderr)
    sys.exit(1)
TARGET.write_text(json.dumps({"language": LANGUAGE, "code": code}))
print(f"[package] packaged {src.name} -> solution.json ({len(code)} chars)")
