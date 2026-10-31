#!/usr/bin/env python3
"""Fail when a workflow reads a GitHub secret. GitHub holds none.

Every secret lives in Google Secret Manager, and a workflow reads it at run
time through Workload Identity (google-github-actions/auth, then
get-secretmanager-secrets). GitHub keeps only identifiers, as variables
(`vars.*`). The rule and the inventory live in the platform repo:
https://github.com/acebot712/promptguard/blob/main/docs/04_engineering/reference/ENVIRONMENT_VARIABLES.md#cicd-only

A copy of promptguard's scripts/utils/check_no_github_secrets.py, kept in step
with it by hand: each of the seven repos runs the same check in its own CI.

The one name allowed is `GITHUB_TOKEN`, which GitHub mints for each run and
nobody stores. Anything else, in an expression (`${{ secrets.X }}`), a
reusable workflow's `secrets:` block, or `secrets: inherit`, is a copy that
would drift from Secret Manager the day someone rotates it there.

Run by CI:

    python scripts/check_no_github_secrets.py --check
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOTS = (Path(".github/workflows"), Path(".github/actions"))
ALLOWED = {"GITHUB_TOKEN"}

# `secrets.NAME` not preceded by a word character or a dot, so
# `steps.secrets.outputs.X` (a step whose id is `secrets`) is not a hit.
EXPRESSION = re.compile(r"(?<![\w.])secrets\.([A-Za-z_][A-Za-z0-9_]*)")
EXPRESSION_INDEX = re.compile(r"(?<![\w.])secrets\[")
INHERIT = re.compile(r"^\s*secrets:\s*inherit\s*$")
# A reusable workflow declaring secrets it accepts: `on: workflow_call: secrets:`.
DECLARED = re.compile(r"^(\s*)secrets:\s*$")


def _in_expression(line: str, start: int) -> bool:
    """True when `start` falls inside a `${{ ... }}` on this line."""
    opened = line.rfind("${{", 0, start)
    return opened != -1 and line.find("}}", opened) >= start


def violations(path: Path) -> list[str]:
    found: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if line.lstrip().startswith("#"):
            continue
        found.extend(
            f"{path}:{number}: secrets.{match.group(1)}"
            for match in EXPRESSION.finditer(line)
            if match.group(1) not in ALLOWED and _in_expression(line, match.start())
        )
        found.extend(
            f"{path}:{number}: secrets[...]"
            for match in EXPRESSION_INDEX.finditer(line)
            if _in_expression(line, match.start())
        )
        if INHERIT.match(line):
            found.append(f"{path}:{number}: secrets: inherit")
        declared = DECLARED.match(line)
        if declared:
            indent = len(declared.group(1))
            for child in lines[number:]:
                if not child.strip() or child.lstrip().startswith("#"):
                    continue
                child_indent = len(child) - len(child.lstrip())
                if child_indent <= indent:
                    break
                name = child.strip().split(":", 1)[0]
                if child_indent == indent + 2 and name not in ALLOWED:
                    found.append(f"{path}:{number}: passes or accepts secret {name}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 on a violation (default)")
    parser.parse_args()

    files = sorted(
        p
        for root in ROOTS
        if root.is_dir()
        for p in root.rglob("*")
        if p.suffix in (".yml", ".yaml")
    )
    found = [v for path in files for v in violations(path)]
    if found:
        print(
            "GitHub holds no secrets: read these from Secret Manager instead",
            file=sys.stderr,
        )
        print(
            "(https://github.com/acebot712/promptguard/blob/main/docs/04_engineering/reference/ENVIRONMENT_VARIABLES.md#cicd-only):",
            file=sys.stderr,
        )
        for v in found:
            print(f"  {v}", file=sys.stderr)
        return 1
    print(
        f"no-github-secrets: {len(files)} workflow file(s) read no GitHub secret but GITHUB_TOKEN."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
