"""
Anthropic SDK patch - wraps ``anthropic.messages.create`` (sync + async).

Covers direct Anthropic SDK usage and frameworks that use it under the
hood (e.g. LangChain's ChatAnthropic).
"""

import importlib.util
import logging
from typing import Any

from promptguard.patches._base import rewrite_message_object, wrap_async, wrap_sync

logger = logging.getLogger("promptguard")

NAME = "anthropic"

_original_sync_create = None
_original_async_create = None
_patched = False


# ---------------------------------------------------------------------------
# Message extraction (Anthropic-specific)
# ---------------------------------------------------------------------------


def _system_to_text(system: Any) -> str | None:
    """Return the guard-message text for ``system``, or ``None`` if it yields
    no message.

    This is the single source of truth for "does ``system`` produce a guard
    message" — used by both extraction and redaction so their indices stay in
    lockstep (a truthy-but-empty ``system`` must not shift the offset).
    """
    if not system:
        return None
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        text_parts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                text_parts.append(block.text)
        if text_parts:
            return "\n".join(text_parts)
    return None


def _tool_result_to_text(content: Any) -> str:
    """Flatten the ``content`` field of a ``tool_result`` block to text.

    Tool results are the canonical indirect prompt-injection channel (text a
    tool fetched from the outside world), so they MUST be scanned.  The field
    is either a plain string or a list of content blocks (text/image); only
    text carries scannable content.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
        return "\n".join(p for p in parts if p)
    return ""


def _flatten_content_blocks(content: Any) -> str:
    """Flatten an Anthropic content-block list to scannable text.

    Includes ``text`` blocks and the text inside ``tool_result`` blocks.
    The whole list still collapses into ONE guard message, so redaction
    indices are unaffected by how many blocks a message contains.
    """
    text_parts = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_result":
                tool_text = _tool_result_to_text(block.get("content"))
                if tool_text:
                    text_parts.append(tool_text)
        elif getattr(block, "type", None) == "tool_result":
            tool_text = _tool_result_to_text(getattr(block, "content", None))
            if tool_text:
                text_parts.append(tool_text)
        elif hasattr(block, "text"):
            text_parts.append(str(block.text))
    return "\n".join(text_parts)


def _messages_to_guard_format(
    messages: Any,
    system: Any = None,
) -> list[dict[str, str]]:
    """Convert Anthropic-style messages to the guard API format.

    Anthropic uses a separate ``system`` parameter (string or list of
    content blocks) rather than a system message in the messages array.
    """
    result: list[dict[str, str]] = []

    system_text = _system_to_text(system)
    if system_text is not None:
        result.append({"role": "system", "content": system_text})

    if not messages:
        return result

    for msg in messages:
        role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "role", "user")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")

        if isinstance(content, list):
            content = _flatten_content_blocks(content)

        result.append({"role": str(role), "content": str(content)})

    return result


def _extract_messages(args, kwargs) -> tuple[list[dict[str, str]], str | None, dict[str, Any]]:
    messages = kwargs.get("messages")
    model = kwargs.get("model")
    system = kwargs.get("system")
    guard_messages = _messages_to_guard_format(messages, system) if messages else []
    return guard_messages, str(model) if model else None, {"framework": "anthropic"}


def _apply_redaction(args, kwargs, redacted: list[dict[str, str]]) -> dict | None:
    """Write redacted content back into Anthropic ``create()`` kwargs.

    Mirrors ``_messages_to_guard_format`` (``system`` first, then every
    message).  Attribute-based message objects are rewritten via a copy; if
    any message with a redacted counterpart cannot be rewritten, returns
    ``None`` so enforce mode escalates to a block.
    """
    if not redacted:
        return None
    new_kwargs: dict = dict(kwargs)
    system = new_kwargs.get("system")
    messages = new_kwargs.get("messages")
    # Use the exact same predicate as extraction so the message index offset
    # matches the redacted_messages list.
    has_system = _system_to_text(system) is not None
    offset = 1 if has_system else 0

    if has_system:
        new_kwargs["system"] = redacted[0]["content"]

    if messages:
        result: list = []
        for i, msg in enumerate(messages):
            idx = i + offset
            if idx >= len(redacted):
                # No redacted counterpart for a scanned message; forwarding
                # the original would leak flagged content.
                return None
            replacement = redacted[idx]["content"]
            if isinstance(msg, dict):
                new_msg = dict(msg)
                # Preserve the block shape: list content is rebuilt as a
                # text block rather than collapsed to a bare string.
                if isinstance(new_msg.get("content"), list):
                    new_msg["content"] = [{"type": "text", "text": replacement}]
                else:
                    new_msg["content"] = replacement
                result.append(new_msg)
            else:
                rewritten = rewrite_message_object(msg, "content", replacement)
                if rewritten is None:
                    return None
                result.append(rewritten)
        new_kwargs["messages"] = result

    return new_kwargs


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
        logger.debug("Failed to extract Anthropic response text", exc_info=True)
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
