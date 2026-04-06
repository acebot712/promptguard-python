# Contributing to PromptGuard Python SDK

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | >= 3.10 | [python.org](https://python.org/) |
| pip | latest | Comes with Python |
| pre-commit | >= 4.0 | `pip install pre-commit` |

## Quick Start

```bash
git clone https://github.com/acebot712/promptguard-python.git
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
```

The project uses [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting. The configuration is in `pyproject.toml` under `[tool.ruff]`.

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

To run manually:

```bash
pre-commit run --all-files
```

## Testing

### Running Tests

```bash
pytest tests/ -v                                          # All tests
pytest tests/ -v --cov --cov-report=term-missing          # With coverage report
pytest tests/ -v --cov --cov-fail-under=50                # With coverage enforcement (CI mode)
```

### Test Files

| File | What it tests |
|---|---|
| `tests/test_client.py` | HTTP client (request building, error handling, retries) |
| `tests/test_guard_client.py` | Guard API (scan, redact, validate-tool) |
| `tests/test_auto_instrumentation.py` | Auto-instrumentation monkey-patching of LLM SDKs |
| `tests/test_integrations.py` | Framework integrations (LangChain, CrewAI, LlamaIndex) |
| `tests/test_contract.py` | Contract tests against `guard-contract.json` |

### Async Tests

The project uses `asyncio_mode = "strict"` (configured in `pyproject.toml`). All async test functions must be explicitly decorated:

```python
import pytest

@pytest.mark.asyncio
async def test_async_scan():
    ...
```

### Contract Tests

`tests/test_contract.py` validates SDK behavior against `tests/guard-contract.json`, a shared contract that defines expected request/response shapes. The Node.js SDK uses the same contract file to ensure both SDKs behave identically.

### Coverage

Coverage is configured in `pyproject.toml`:

- Source: `promptguard/` package
- Minimum threshold: **50%** (`fail_under = 50`)
- Shows missing lines in terminal output

### Environment Variables

Tests mock the API by default. To run against a live API:

```bash
PROMPTGUARD_API_KEY=pg_test_... PROMPTGUARD_BASE_URL=http://localhost:8080 pytest tests/ -v
```

| Variable | Default | Description |
|---|---|---|
| `PROMPTGUARD_API_KEY` | (none) | API key for live testing |
| `PROMPTGUARD_BASE_URL` | `https://api.promptguard.co` | API base URL |

## CI/CD

CI runs on every push to `main` and on PRs (`.github/workflows/ci.yml`):

| Job | What it does |
|---|---|
| **Lint & Format** | `ruff check .` + `ruff format --check .` (using ruff 0.15.4) |
| **Test** | `pytest` with coverage on Python 3.10, 3.11, 3.12, 3.13 |
| **Build** | `python -m build` + verify import works |

Reproduce CI locally:

```bash
ruff check . && ruff format --check . && pytest tests/ -v --cov --cov-fail-under=50
```

> **Note:** CI pins `ruff==0.15.4`. The dev dependency in `pyproject.toml` pins `ruff==0.15.8`. Both should work, but if you see lint differences, check your local ruff version.

## Releasing

Releases are triggered by creating a GitHub Release:

1. Update `version` in `pyproject.toml`
2. Commit and push to `main`
3. Create a GitHub Release

The release workflow (`.github/workflows/release.yml`):

1. Validates (ruff lint, ruff format, pytest, build)
2. Checks if the version already exists on PyPI
3. Publishes via PyPI Trusted Publishing (OIDC, no token needed)

Trusted Publishing requires the PyPI project to have a configured trusted publisher for `acebot712/promptguard-python` with workflow `release.yml`.

## PR Checklist

- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] `pytest tests/ -v --cov --cov-fail-under=50` passes
- [ ] `pre-commit run --all-files` passes
- [ ] New functionality has tests
- [ ] Contract tests updated if request/response shapes changed
- [ ] PR description explains the change

## Reporting Issues

Open an issue at https://github.com/acebot712/promptguard-python/issues with:

- Python version (`python --version`)
- SDK version (`pip show promptguard-sdk`)
- Minimal reproduction steps
