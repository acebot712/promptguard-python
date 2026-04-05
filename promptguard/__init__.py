"""
PromptGuard Python SDK

Drop-in security for AI applications.
Just change your base URL and add an API key.

Usage (proxy mode - existing):

    from promptguard import PromptGuard

    pg = PromptGuard(api_key="pg_xxx")
    response = pg.chat.completions.create(
        model="gpt-4",
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

from promptguard.auto import init, shutdown
from promptguard.client import PromptGuard, PromptGuardAsync
from promptguard.config import Config
from promptguard.guard import GuardApiError, GuardClient, GuardDecision, PromptGuardBlockedError

__version__ = "1.5.3"
__all__ = [
    "Config",
    "GuardApiError",
    "GuardClient",
    "GuardDecision",
    "PromptGuard",
    "PromptGuardAsync",
    "PromptGuardBlockedError",
    "init",
    "shutdown",
]
