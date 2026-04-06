"""Shared credential resolution for PromptGuard components."""

import os

_DEFAULT_BASE_URL = "https://api.promptguard.co/api/v1"


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
            "PromptGuard API key required. Pass api_key= or set the "
            "PROMPTGUARD_API_KEY environment variable."
        )
    url = base_url or os.environ.get("PROMPTGUARD_BASE_URL") or default_base_url
    return key, url
