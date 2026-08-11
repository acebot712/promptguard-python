#!/usr/bin/env python3
"""The version being released must have a CHANGELOG entry.

A CHANGELOG is only worth having if it is written when the release is cut. Left
to be reconstructed afterwards it becomes a list of tag subjects -- which is
exactly what the entries below 1.11.1 in this file are, because there was no
CHANGELOG until 2026-08-11 and the tag subject was all anyone had recorded.

So this runs in the release workflow, before publishing. It is deliberately dumb:
it does not check the prose, only that somebody wrote *something* under the
version about to ship. A gate that tried to judge quality would be argued with;
one that asks "is there an entry" is either satisfied or it is not.

    python scripts/check_changelog.py           # check the current version
    python scripts/check_changelog.py 1.12.0    # check a specific one
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
VERSION_FILE = ROOT / "promptguard" / "_version.py"


def current_version() -> str | None:
    if not VERSION_FILE.exists():
        return None
    match = re.search(r"(\d+\.\d+\.\d+)", VERSION_FILE.read_text())
    return match.group(1) if match else None


def main() -> int:
    version = sys.argv[1] if len(sys.argv) > 1 else current_version()
    if not version:
        print("error: could not determine the version to check", file=sys.stderr)
        return 1

    if not CHANGELOG.exists():
        print(f"error: {CHANGELOG.name} does not exist", file=sys.stderr)
        return 1

    text = CHANGELOG.read_text()
    # `## [1.12.0]` with any suffix (a date, or nothing yet).
    if not re.search(rf"^##\s*\[{re.escape(version)}\]", text, re.M):
        print(
            f"error: no CHANGELOG entry for {version}.\n"
            f"       Add a '## [{version}] — YYYY-MM-DD' section to {CHANGELOG.name} "
            "describing what changed for the people installing it.\n"
            "       Reconstructing this after the fact produces a list of commit "
            "subjects, which is what the pre-1.11.1 entries already are.",
            file=sys.stderr,
        )
        return 1

    print(f"✓ CHANGELOG has an entry for {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
