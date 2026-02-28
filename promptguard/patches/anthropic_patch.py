"""
Anthropic SDK patch — wraps ``anthropic.messages.create`` (sync + async).

Covers direct Anthropic SDK usage and frameworks that use it under the
hood (e.g. LangChain's ChatAnthropic).
"""

import importlib.util
import logging
from typing import Any

from promptguard.patches._base import wrap_async, wrap_sync

logger = logging.getLogger("promptguard")

NAME = "anthropic"

_original_sync_create = None
_original_async_create = None
_patched = False


# ---------------------------------------------------------------------------
# Message extraction (Anthropic-specific)
# ---------------------------------------------------------------------------


def _messages_to_guard_format(
    messages: Any,
    system: Any = None,
) -> list[dict[str, str]]:
    """Convert Anthropic-style messages to the guard API format.

    Anthropic uses a separate ``system`` parameter (string or list of
    content blocks) rather than a system message in the messages array.
    """
    result: list[dict[str, str]] = []

    if system:
        if isinstance(system, str):
            result.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text_parts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif hasattr(block, "text"):
                    text_parts.append(block.text)
            if text_parts:
                result.append({"role": "system", "content": "\n".join(text_parts)})

    if not messages:
        return result

    for msg in messages:
        role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "role", "user")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")

        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif hasattr(block, "text"):
                    text_parts.append(block.text)
            content = "\n".join(text_parts)

        result.append({"role": str(role), "content": str(content)})

    return result


def _extract_messages(args, kwargs) -> tuple[list[dict[str, str]], str | None, dict[str, Any]]:
    messages = kwargs.get("messages")
    model = kwargs.get("model")
    system = kwargs.get("system")
    guard_messages = _messages_to_guard_format(messages, system) if messages else []
    return guard_messages, str(model) if model else None, {"framework": "anthropic"}


def _apply_redaction(args, kwargs, redacted: list[dict[str, str]]) -> dict:
    kwargs = dict(kwargs)
    system = kwargs.get("system")
    messages = kwargs.get("messages")
    has_system = system is not None
    offset = 1 if has_system else 0

    if has_system and redacted:
        kwargs["system"] = redacted[0]["content"]

    if messages and redacted:
        result = []
        for i, msg in enumerate(messages):
            idx = i + offset
            if idx < len(redacted):
                if isinstance(msg, dict):
                    new_msg = dict(msg)
                    new_msg["content"] = redacted[idx]["content"]
                    result.append(new_msg)
                else:
                    result.append(msg)
            else:
                result.append(msg)
        kwargs["messages"] = result

    return kwargs


def _extract_response_content(response: Any) -> str | None:
    """Extract text content from an Anthropic Message response."""
    try:
        if hasattr(response, "content") and response.content:
            parts = []
            for block in response.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts) if parts else None
        if isinstance(response, dict):
            content = response.get("content", [])
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts) if parts else None
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Apply / revert
# ---------------------------------------------------------------------------


def apply() -> bool:
    global _original_sync_create, _original_async_create, _patched

    if _patched:
        return True

    if importlib.util.find_spec("anthropic") is None:
        return False

    try:
        from anthropic.resources.messages import Messages

        _original_sync_create = Messages.create
        Messages.create = wrap_sync(
            Messages.create,
            _extract_messages,
            _extract_response_content,
            _apply_redaction,
        )
    except (ImportError, AttributeError):
        logger.debug("Could not patch anthropic sync Messages.create")

    try:
        from anthropic.resources.messages import AsyncMessages

        _original_async_create = AsyncMessages.create
        AsyncMessages.create = wrap_async(
            AsyncMessages.create,
            _extract_messages,
            _extract_response_content,
            _apply_redaction,
        )
    except (ImportError, AttributeError):
        logger.debug("Could not patch anthropic async AsyncMessages.create")

    _patched = _original_sync_create is not None or _original_async_create is not None
    return _patched


def revert() -> None:
    global _original_sync_create, _original_async_create, _patched

    if not _patched:
        return

    try:
        if _original_sync_create:
            from anthropic.resources.messages import Messages

            Messages.create = _original_sync_create
        if _original_async_create:
            from anthropic.resources.messages import AsyncMessages

            AsyncMessages.create = _original_async_create
    except (ImportError, AttributeError):
        pass

    _original_sync_create = None
    _original_async_create = None
    _patched = False
