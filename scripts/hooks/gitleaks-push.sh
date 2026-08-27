#!/usr/bin/env bash
#
# Secret-scan the commits a push would add.
#
# WHY THE SCOPE IS THE PUSHED RANGE
# ---------------------------------
# Scanning the whole repository or its history fails on the first run and keeps
# failing: this codebase carries secret-SHAPED strings on purpose, in detector
# fixtures and eval corpora, and history cannot be fixed by the commit that
# trips over it. A gate that always fails teaches `--no-verify`, which is worse
# than no gate because you then believe you are covered. Scanning only what the
# push adds is fast and every finding is actionable.
#
# Known fixture paths are allowlisted in .gitleaks.toml where this repo has
# them. Add to that file rather than bypassing the hook.
set -euo pipefail

command -v gitleaks >/dev/null 2>&1 || {
  echo "gitleaks not installed; skipping the local secret scan." >&2
  echo "  brew install gitleaks" >&2
  exit 0
}

# Set by pre-commit at the pre-push stage. Falling back to "everything not yet
# on a remote" covers a direct hook invocation and a brand-new branch alike.
from_ref="${PRE_COMMIT_FROM_REF:-}"
to_ref="${PRE_COMMIT_TO_REF:-}"
if [ -n "$from_ref" ] && ! git rev-parse --quiet --verify "$from_ref" >/dev/null 2>&1; then
  from_ref=""
fi

if [ -n "$from_ref" ] && [ -n "$to_ref" ]; then
  log_opts="$from_ref..$to_ref"
else
  log_opts="HEAD --not --remotes"
fi

if ! gitleaks detect --source . --log-opts "$log_opts" --no-banner --redact; then
  cat >&2 <<'MSG'

A secret was found in the commits you are pushing.

This repository is public. A credential that lands on the default branch is
world-readable immediately and must be rotated, not just reverted.

If this is a test fixture, do not bypass -- give it an obviously-fake shape, or
add its path to .gitleaks.toml so the exemption is written down and reviewable.

Bypass (last resort): git push --no-verify
MSG
  exit 1
fi
