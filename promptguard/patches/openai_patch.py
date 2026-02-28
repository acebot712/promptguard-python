"""
OpenAI SDK patch — wraps ``openai.chat.completions.create`` (sync + async).

This single patch covers every framework built on top of the OpenAI Python
SDK, including LangChain (ChatOpenAI), CrewAI, AutoGen, Semantic Kernel,
and direct usage.
"""

import importlib.util
import logging
from typing import Any

from promptguard.patches._base import wrap_async, wrap_sync

logger = logging.getLogger("promptguard")

NAME = "openai"

_original_sync_create = None
_original_async_create = None
_patched = False


# ---------------------------------------------------------------------------
# Message extraction / redaction (OpenAI-specific)
# ---------------------------------------------------------------------------


def _messages_to_guard_format(messages: Any) -> list[dict[str, str]]:
    """Convert OpenAI-style messages to the guard API format."""
    result = []
    if not messages:
        return result
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                content = "\n".join(text_parts)
            result.append({"role": msg.get("role", "user"), "content": str(content)})
        elif hasattr(msg, "role") and hasattr(msg, "content"):
            content = msg.content or ""
            if isinstance(content, list):
                text_parts = [
                    part.text if hasattr(part, "text") else part.get("text", "")
                    for part in content
                    if hasattr(part, "text")
                    or (isinstance(part, dict) and part.get("type") == "text")
                ]
                content = "\n".join(text_parts)
            result.append({"role": str(msg.role), "content": str(content)})
    return result


def _extract_messages(args, kwargs) -> tuple[list[dict[str, str]], str | None, dict[str, Any]]:
    """Extract messages and model from OpenAI create() call signature."""
    messages = kwargs.get("messages") or (args[1] if len(args) > 1 else None)
    model = kwargs.get("model") or (args[0] if args else None)
    guard_messages = _messages_to_guard_format(messages) if messages else []
    return guard_messages, str(model) if model else None, {"framework": "openai"}


def _apply_redaction(args, kwargs, redacted: list[dict[str, str]]) -> dict:
    """Apply redacted content back into kwargs."""
    messages = kwargs.get("messages") or (args[1] if len(args) > 1 else None)
    if not messages or not redacted:
        return kwargs
    result = []
    for i, msg in enumerate(messages):
        if i < len(redacted):
            if isinstance(msg, dict):
                new_msg = dict(msg)
                new_msg["content"] = redacted[i]["content"]
                result.append(new_msg)
            else:
                result.append(msg)
        else:
            result.append(msg)
    kwargs = dict(kwargs)
    kwargs["messages"] = result
    return kwargs


def _extract_response_content(response: Any) -> str | None:
    """Extract text content from an OpenAI ChatCompletion response."""
    try:
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                return choice.message.content
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content")
    except Exception:
        logger.debug("Failed to extract OpenAI response text", exc_info=True)
    return None


# ---------------------------------------------------------------------------
# Apply / revert
# ---------------------------------------------------------------------------


def apply() -> bool:
    """Patch the OpenAI SDK. Returns True if successfully patched."""
    global _original_sync_create, _original_async_create, _patched

    if _patched:
        return True

    if importlib.util.find_spec("openai") is None:
        return False

    try:
        from openai.resources.chat.completions import Completions

        _original_sync_create = Completions.create
        Completions.create = wrap_sync(
            Completions.create,
            _extract_messages,
            _extract_response_content,
            _apply_redaction,
        )
    except (ImportError, AttributeError):
        logger.debug("Could not patch openai sync Completions.create")

    try:
        from openai.resources.chat.completions import AsyncCompletions

        _original_async_create = AsyncCompletions.create
        AsyncCompletions.create = wrap_async(
            AsyncCompletions.create,
            _extract_messages,
            _extract_response_content,
            _apply_redaction,
        )
    except (ImportError, AttributeError):
        logger.debug("Could not patch openai async AsyncCompletions.create")

    _patched = _original_sync_create is not None or _original_async_create is not None
    return _patched


def revert() -> None:
    """Remove the OpenAI SDK patch."""
    global _original_sync_create, _original_async_create, _patched

    if not _patched:
        return

    try:
        if _original_sync_create:
            from openai.resources.chat.completions import Completions

            Completions.create = _original_sync_create

        if _original_async_create:
            from openai.resources.chat.completions import AsyncCompletions

            AsyncCompletions.create = _original_async_create
    except (ImportError, AttributeError):
        pass

    _original_sync_create = None
    _original_async_create = None
    _patched = False
