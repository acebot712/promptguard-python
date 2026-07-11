"""Shared credential resolution for PromptGuard components."""

import os

_DEFAULT_BASE_URL = "https://api.promptguard.co/api/v1"

# The only valid enforcement modes. Kept here (not duplicated per integration)
# so every entry point validates identically.
VALID_MODES = ("enforce", "monitor")


def validate_mode(mode: str) -> str:
    """Validate an enforcement ``mode``, raising ``ValueError`` if unknown.

    Every entry point (``init()`` and each framework integration) must call
    this. A silently-accepted bad mode (e.g. ``"Enforce"``, ``"block"``, a
    typo) is security-adjacent: integrations only *block* when the mode is
    exactly ``"enforce"``, so an unrecognised value would fail open and stop
    blocking. Fail loudly at construction time instead.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be 'enforce' or 'monitor', got {mode!r}")
    return mode


def resolve_credentials(
    api_key: str | None = None,
    base_url: str | None = None,
    default_base_url: str = _DEFAULT_BASE_URL,
) -> tuple[str, str]:
    """Resolve API key and base URL from explicit params or environment.

    Returns ``(api_key, base_url)``; raises ``ValueError`` if no key is found.
    """
    key = api_key or os.environ.get("PROMPTGUARD_API_KEY", "")
    if not key:
        raise ValueError(
            "API key required. Pass api_key parameter or set the "
            "PROMPTGUARD_API_KEY environment variable. "
            "Get a key at https://app.promptguard.co"
        )
    url = base_url or os.environ.get("PROMPTGUARD_BASE_URL") or default_base_url
    return key, url
