# AGENTS.md

## Overview

Python SDK for PromptGuard (`promptguard-sdk` on PyPI). Provides auto-instrumentation for LLM SDKs (OpenAI, Anthropic, Google, Cohere, Bedrock), a Guard API client, and framework integrations (LangChain, CrewAI, LlamaIndex).

## Repository Layout

```
promptguard/
├── client.py          # HTTP client (proxy mode)
├── guard.py           # Guard API client
├── auto.py            # Auto-instrumentation (promptguard.init())
├── config.py          # Configuration and environment
├── patches/           # SDK monkey-patches (openai, anthropic, etc.)
└── integrations/      # Framework integrations (langchain, crewai, etc.)

tests/
├── test_client.py
├── test_guard_client.py
├── test_auto_instrumentation.py
├── test_integrations.py
├── test_contract.py
└── guard-contract.json    # Shared contract spec
```

## Setup

```bash
pip install -e ".[dev]"
pre-commit install               # Optional: enable hooks
```

## Building and Testing

```bash
pytest tests/ -v                              # Run all tests
pytest tests/ -v --cov --cov-report=term-missing  # With coverage
pytest tests/test_client.py -v               # Single file
pre-commit run --all-files                   # All hooks

# Live API tests (optional, needs real credentials)
PROMPTGUARD_API_KEY=pg_... PROMPTGUARD_BASE_URL=https://api.promptguard.co pytest tests/ -v
```

## Code Quality

```bash
ruff check .                    # Lint
ruff check . --fix              # Lint with auto-fix
ruff format .                   # Format
ruff format . --check           # Check formatting
```

Always run `ruff check` and `ruff format` after editing Python files.

## Coding Standards

- Python 3.10+ required
- Type hints on all public functions and methods
- Ruff for both linting and formatting (configured in `pyproject.toml`)
- Tests use pytest with pytest-asyncio for async code
- Avoid adding runtime dependencies without discussion
- Keep the public API surface small: `init()`, `GuardClient`, and framework integrations

## Commit Messages

- Imperative mood: "Add X" not "Added X"
- Focus on what changed from the user's perspective
- Reference issues when applicable

## Boundaries

### Never
- Commit API keys, tokens, or credentials
- Add heavyweight runtime dependencies (the SDK should stay lightweight)
- Break the public API without a major version bump
- Modify `guard-contract.json` without coordinating with the Node.js SDK
