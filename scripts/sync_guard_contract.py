#!/usr/bin/env python3
"""
Adopt a freshly-fetched cross-SDK contract into tests/, and stamp its lockfile.

WHY THIS EXISTS
---------------
`tests/guard-contract.json` is the cross-SDK contract: this SDK and the Node SDK
must both satisfy every case in it. Its source of truth is the platform
monorepo's `packages/sdk-shared/guard-contract.json`, and for a long time the
only thing connecting the two was somebody remembering to copy the file.

Nobody did. On 2026-08-11 the monorepo source was found two minor versions
behind this copy (v1.3.0 against v1.5.1) and missing an entire
`redaction_enforcement` section. Five months, undetected -- because
`tests/test_contract.py` opens with "if this test fails, the Python SDK has
drifted from the cross-SDK contract" while reading a local duplicate of itself.
A file compared against its own copy cannot detect drift.

The monorepo now publishes the contract at a public URL, and
`.github/workflows/sync-from-api.yml` fetches it weekly beside the OpenAPI
spec. This script is the "adopt" half of that: it writes the fetched contract
over the vendored copy and records where it came from in
`tests/guard-contract.lock.json`.

The lockfile is what gives `tests/test_contract.py` something external to check
against. Hand-editing `tests/guard-contract.json` now fails the suite, because
the digest no longer matches the one recorded at sync time. Both files move
together only through this script, and only inside a reviewed PR -- the
workflow never pushes to a branch anyone merges from.

USAGE
    python scripts/sync_guard_contract.py <fetched-contract.json>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "tests" / "guard-contract.json"
LOCK = REPO / "tests" / "guard-contract.lock.json"

# Where the monorepo publishes it. Repeated in sync-from-api.yml (the fetch)
# and asserted in tests/test_contract.py, so moving it is a three-line change
# that shows up in review rather than a silent redirect.
SOURCE_URL = "https://promptguard.co/contracts/guard-contract.json"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write(f"{__doc__}\n")
        return 2

    fetched = Path(argv[1])
    if not fetched.is_file():
        sys.stderr.write(f"No such file: {fetched}\n")
        return 1

    raw = fetched.read_bytes()

    # Parse before adopting. A 200 carrying an HTML error page would otherwise
    # be written straight over the contract and only fail later, somewhere less
    # obvious.
    try:
        contract = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"Aborting: {fetched} is not JSON ({exc}); refusing to adopt\n")
        return 1

    version = contract.get("_version")
    if not version:
        sys.stderr.write(f"Aborting: {fetched} carries no _version; refusing to adopt\n")
        return 1

    digest = hashlib.sha256(raw).hexdigest()

    CONTRACT.write_bytes(raw)
    LOCK.write_text(
        json.dumps(
            {
                "_comment": (
                    "Provenance for tests/guard-contract.json. Written by "
                    "scripts/sync_guard_contract.py from "
                    ".github/workflows/sync-from-api.yml. Do not hand-edit: "
                    "tests/test_contract.py checks the contract against the "
                    "digest recorded here, so editing either file alone fails "
                    "the suite -- which is the drift this pair exists to catch."
                ),
                "source": SOURCE_URL,
                "version": version,
                "sha256": digest,
            },
            indent=2,
        )
        + "\n"
    )

    sys.stderr.write(f"Adopted cross-SDK contract v{version} → sha256 {digest}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
