# Contributing to PromptGuard Python SDK

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | >= 3.10 | [python.org](https://python.org/) |
| pip | latest | Comes with Python |
| pre-commit | >= 4.0 | `pip install pre-commit` |

## Quick Start

```bash
git clone https://github.com/promptguardhq/promptguard-python.git
cd promptguard-python
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install          # Install pre-commit hooks
pytest tests/ -v            # Run tests
```

## Development Workflow

### Code Quality

```bash
ruff check .                # Lint
ruff check . --fix          # Lint and auto-fix
ruff format .               # Format
ruff format --check .       # Check formatting without fixing
mypy promptguard            # Type check (strict settings in [tool.mypy])
```

The project uses [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting. The configuration is in `pyproject.toml` under `[tool.ruff]`. The one ruff version is the `[dev]` pin in `pyproject.toml`; CI installs exactly that pin, and the pre-commit hook's `rev` matches it, so the hook, your shell and CI format identically. Bump all three together.

### Pre-commit Hooks

Pre-commit runs automatically on every commit after `pre-commit install`. Hooks include:

- Trailing whitespace removal
- End-of-file fixer
- Merge conflict detection
- YAML validation
- Large file check
- Private key detection
- Ruff lint (with auto-fix)
- Ruff format
- gitleaks secret scan of the commits being pushed (a pre-push hook, installed by the same `pre-commit install`)

To run manually:

```bash
pre-commit run --all-files
```

## Testing

### Running Tests

```bash
pytest tests/ -v                                          # All tests
pytest tests/ -v --cov --cov-report=term-missing          # With coverage report
pytest tests/ -v --cov --cov-fail-under=60                # With coverage enforcement (CI mode)
```

### Test Files

One `tests/test_<area>.py` per module or integration (`test_client`, `test_guard_client`, `test_auto_instrumentation`, `test_integrations`, `test_langchain_core`, `test_fastapi_middleware`, ...). `tests/validate_openapi.py` is not a pytest file: CI runs it against the live developer OpenAPI spec.

### Async Tests

The project uses `asyncio_mode = "strict"` (configured in `pyproject.toml`). All async test functions must be explicitly decorated:

```python
import pytest


@pytest.mark.asyncio
async def test_async_scan(): ...
```

### Contract Tests

`tests/test_contract.py` validates SDK behavior against `tests/guard-contract.json`, a shared contract that defines expected request/response shapes. The Node.js SDK uses the same contract file to ensure both SDKs behave identically.

### Coverage

Coverage is configured in `pyproject.toml`:

- Source: `promptguard/` package
- Minimum threshold: **60%** (`fail_under = 60`)
- Shows missing lines in terminal output

### No live-API mode

The whole suite is mocked: no test makes a network call. Setting `PROMPTGUARD_API_KEY` or `PROMPTGUARD_BASE_URL` does not point any test at a real API; the tests that depend on them set or delete them with `monkeypatch`, so values in your shell cannot change the result. To try the SDK against a real backend, write a throwaway script.

## CI/CD

CI runs on every push to `main` and on PRs (`.github/workflows/ci.yml`):

| Job | What it does |
|---|---|
| **Lint & Format** | `ruff check .` + `ruff format --check .`, using the `[dev]` ruff pin read from `pyproject.toml` |
| **Type Check** | `mypy promptguard` |
| **Test** | `pytest` with coverage (`--cov-fail-under=60`) on Python 3.10, 3.11, 3.12, 3.13, 3.14 |
| **LangChain extras floor** | The LangChain tests on Python 3.10 against the lowest `langchain-core` the `all` extra allows |
| **Validate API Contract** | Downloads the developer OpenAPI spec and checks SDK types against it |
| **Build** | `python -m build` + verify import works |

Secret scanning (gitleaks) runs on every push and PR in `.github/workflows/secret-scan.yml`, kept out of `ci.yml` so docs-only commits are still scanned.

Reproduce CI locally:

```bash
ruff check . && ruff format --check . && mypy promptguard && pytest tests/ -v --cov --cov-fail-under=60
```

## Releasing

Releases are triggered by creating a GitHub Release:

1. Update `__version__` in `promptguard/_version.py` — `pyproject.toml` reads it
   from there (`[tool.hatch.version]`) and has no version of its own to edit
2. Commit and push to `main`
3. Create a GitHub Release

The release workflow (`.github/workflows/release.yml`):

1. Validates (ruff lint, ruff format, pytest, build)
2. Checks if the version already exists on PyPI
3. Publishes via PyPI Trusted Publishing (OIDC, no token needed)

Trusted Publishing requires the PyPI project to have a configured trusted publisher for `promptguardhq/promptguard-python` with workflow `release.yml`.

## PR Checklist

- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] `mypy promptguard` passes
- [ ] `pytest tests/ -v --cov --cov-fail-under=60` passes
- [ ] `pre-commit run --all-files` passes
- [ ] New functionality has tests
- [ ] Contract tests updated if request/response shapes changed
- [ ] PR description explains the change

## Reporting Issues

Open an issue at https://github.com/promptguardhq/promptguard-python/issues with:

- Python version (`python --version`)
- SDK version (`pip show promptguard-sdk`)
- Minimal reproduction steps
