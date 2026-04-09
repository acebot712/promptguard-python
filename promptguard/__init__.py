"""
PromptGuard Python SDK

Drop-in security for AI applications.
Just change your base URL and add an API key.

Usage (proxy mode - existing):

    from promptguard import PromptGuard

    pg = PromptGuard(api_key="pg_xxx")
    response = pg.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Hello!"}]
    )

Usage (auto-instrumentation - new):

    import promptguard
    promptguard.init(api_key="pg_xxx")

    # Now ALL LLM calls are secured, regardless of framework.
    from openai import OpenAI
    client = OpenAI()
    client.chat.completions.create(...)  # scanned by PromptGuard
"""

from promptguard._version import __version__
from promptguard.auto import init, shutdown
from promptguard.client import PromptGuard, PromptGuardAsync, PromptGuardError
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
    "__version__",
    "init",
    "shutdown",
]
