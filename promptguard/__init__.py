"""
PromptGuard Python SDK

Drop-in security for AI applications.
Just change your base URL and add an API key.

Usage (proxy mode - existing):

    from promptguard import PromptGuard

    pg = PromptGuard(api_key="pg_live_xxx")
    response = pg.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Hello!"}]
    )

Usage (auto-instrumentation - new):

    import promptguard
    promptguard.init(api_key="pg_live_xxx")

    # Now ALL LLM calls are secured, regardless of framework.
    from openai import OpenAI
    client = OpenAI()
    client.chat.completions.create(...)  # scanned by PromptGuard
"""

from promptguard._version import __version__
from promptguard.auto import init, is_active, patched_sdks, shutdown
from promptguard.client import (
    PromptGuard,
    PromptGuardAsync,
    PromptGuardError,
    SecurityScanResult,
)
from promptguard.config import Config
from promptguard.guard import GuardApiError, GuardClient, GuardDecision, PromptGuardBlockedError

__all__ = [
    "Config",
    "GuardApiError",
    "GuardClient",
    "GuardDecision",
    "PromptGuard",
    "PromptGuardAsync",
    "PromptGuardBlockedError",
    "PromptGuardError",
    "SecurityScanResult",
    "__version__",
    "init",
    "is_active",
    "patched_sdks",
    "shutdown",
]
