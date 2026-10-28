# AGENTS.md

Python SDK for PromptGuard, published to PyPI as `promptguard-sdk`; the Node SDK
(`promptguard-node`) is its deliberate mirror. Domain vocabulary lives in
`CONTEXT.md`; layout, dependencies and tool config live in `pyproject.toml`.

## Setup

```bash
pip install -e ".[dev]"
pre-commit install               # once per clone: wires the commit AND push hooks
```

Keep API keys, tokens and credentials out of every commit. The repo is public
and on PyPI, so a credential that reaches `main` is world-readable at once and
has to be rotated, not reverted. The push hook runs gitleaks over the commits
being pushed; CI's `secret-scan.yml` is the backstop. A fixture that must look
like a credential gets a path exemption in `.gitleaks.toml`, where it is
reviewable, instead of a `--no-verify`. Why the scan covers only the pushed
range: `scripts/hooks/gitleaks-push.sh`.

## Checks

After editing Python, run `ruff check --fix .` and `ruff format .`. Before
pushing, reproduce CI (`.github/workflows/ci.yml`):

```bash
ruff check . && ruff format --check . && mypy promptguard && pytest tests/ --cov
```

The suite is fully mocked and offline. `pytest` turns warnings into errors and
runs pytest-asyncio in strict mode, so every async test needs
`@pytest.mark.asyncio`. Code must run on Python 3.10, the oldest CI leg.

## Coding standards

- Type hints on every public function and method.
- The SDK stays lightweight: a new runtime dependency needs discussion first.
- The public surface is `__all__` in `promptguard/__init__.py`; keep it small,
  and change or remove a name there only with a major version bump.

## The guard contract and generated types

`tests/guard-contract.json` holds the cases both SDKs must pass, and
`promptguard/generated/` is built from the published OpenAPI spec. Both arrive
through the scheduled `sync-from-api.yml` PR; hand-edits fail the suite (the
contract lock) or get overwritten (the types). A behaviour change that both
SDKs share starts upstream in the contract. Before touching either, read
`docs/adr/0001-shared-guard-contract.md`.

## Commit messages

Conventional prefix plus imperative mood (`fix: drop a session id over 255
characters`), describing the change as a user sees it, with the issue reference
where there is one (`(promptguard#956)`, `(#945)`).

## Agent skills

### Issue tracker

Issues live in this repo's GitHub Issues, worked via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, each label string equal to its name. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` plus `docs/adr/` at the repo root; add to them as terms and decisions settle. See `docs/agents/domain.md`.
