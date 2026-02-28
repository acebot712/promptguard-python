# Contributing to PromptGuard Python SDK

Thank you for your interest in contributing to PromptGuard!

## Development Setup

```bash
git clone https://github.com/acebot712/promptguard-python.git
cd promptguard-python
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Code Quality

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check .
ruff format .
```

## Running Tests

```bash
pytest tests/ -v
```

## Pull Requests

1. Fork the repo and create a feature branch from `main`.
2. Write tests for any new functionality.
3. Ensure `ruff check .` and `ruff format --check .` pass with zero errors.
4. Open a PR with a clear description of the change.

## Reporting Issues

Open an issue at https://github.com/acebot712/promptguard-python/issues with:
- Python version
- SDK version (`pip show promptguard-sdk`)
- Minimal reproduction steps
