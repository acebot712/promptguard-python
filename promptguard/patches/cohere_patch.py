"""
Cohere SDK patch - wraps ``cohere.Client.chat`` / ``ClientV2.chat``
(sync + async).

Covers direct Cohere usage and frameworks that use it (Haystack, LangChain).
"""

import importlib.util
import logging
from typing import Any

from promptguard.patches._base import rewrite_message_object, wrap_async, wrap_sync

logger = logging.getLogger("promptguard")

NAME = "cohere"

_originals: dict[str, Any] = {}
_patched = False


# ---------------------------------------------------------------------------
# Message extraction (Cohere-specific)
# ---------------------------------------------------------------------------


def _to_guard_messages(
    message: Any = None,
    chat_history: Any = None,
    messages: Any = None,
) -> list[dict[str, str]]:
    """Convert Cohere chat args to guard API format."""
    result: list[dict[str, str]] = []

    if messages:
        for msg in messages:
            if isinstance(msg, dict):
                result.append(
                    {
                        "role": msg.get("role", "user"),
                        "content": str(msg.get("content", "")),
                    }
                )
            elif hasattr(msg, "role") and hasattr(msg, "content"):
                result.append({"role": str(msg.role), "content": str(msg.content or "")})
        return result

    if chat_history:
        for msg in chat_history:
            if isinstance(msg, dict):
                result.append(
                    {
                        "role": msg.get("role", "user"),
                        "content": str(msg.get("message", msg.get("content", ""))),
                    }
                )
            elif hasattr(msg, "role"):
                content = getattr(msg, "message", getattr(msg, "content", ""))
                result.append({"role": str(msg.role), "content": str(content or "")})

    if message:
        result.append({"role": "user", "content": str(message)})

    return result


def _extract_messages(args, kwargs) -> tuple[list[dict[str, str]], str | None, dict[str, Any]]:
    guard_messages = _to_guard_messages(
        message=kwargs.get("message"),
        chat_history=kwargs.get("chat_history"),
        messages=kwargs.get("messages"),
    )
    model = kwargs.get("model")
    return guard_messages, str(model) if model else "cohere", {"framework": "cohere"}


def _emits_v2_guard_message(msg: Any) -> bool:
    """Whether ``_to_guard_messages`` emits a guard message for a ``messages``
    (v2) entry.  Single source of truth for extraction and redaction so their
    indices stay aligned (skipped entries must not consume a redacted one)."""
    return isinstance(msg, dict) or (hasattr(msg, "role") and hasattr(msg, "content"))


def _emits_history_guard_message(msg: Any) -> bool:
    """Same as ``_emits_v2_guard_message`` but for ``chat_history`` entries."""
    return isinstance(msg, dict) or hasattr(msg, "role")


def _apply_redaction(args, kwargs, redacted: list[dict[str, str]]) -> dict | None:
    """Write redacted content back into Cohere chat kwargs.

    Mirrors the extraction order in ``_to_guard_messages``:
    - ``messages`` (v2) takes precedence; or
    - ``chat_history`` entries first, then the trailing ``message`` string.

    Only entries that emitted a guard message consume a redacted message, so
    the indices stay aligned with extraction.  Attribute-based message
    objects (e.g. ``cohere.ChatMessage``) are rewritten via a copy; if any
    message with a redacted counterpart cannot be rewritten, returns ``None``
    so enforce mode escalates to a block.
    """
    if not redacted:
        return None
    new_kwargs: dict = dict(kwargs)

    messages = new_kwargs.get("messages")
    if messages:
        rebuilt: list = []
        guard_idx = 0
        for msg in messages:
            if not _emits_v2_guard_message(msg):
                rebuilt.append(msg)
                continue
            replacement = redacted[guard_idx] if guard_idx < len(redacted) else None
            guard_idx += 1
            if replacement is None:
                # No redacted counterpart for a scanned message; forwarding
                # the original would leak flagged content.
                return None
            if isinstance(msg, dict):
                new_msg = dict(msg)
                new_msg["content"] = replacement["content"]
                rebuilt.append(new_msg)
            else:
                rewritten = rewrite_message_object(msg, "content", replacement["content"])
                if rewritten is None:
                    return None
                rebuilt.append(rewritten)
        new_kwargs["messages"] = rebuilt
        return new_kwargs

    guard_idx = 0
    chat_history = new_kwargs.get("chat_history")
    if chat_history:
        rebuilt = []
        for msg in chat_history:
            if not _emits_history_guard_message(msg):
                rebuilt.append(msg)
                continue
            replacement = redacted[guard_idx] if guard_idx < len(redacted) else None
            guard_idx += 1
            if replacement is None:
                return None
            if isinstance(msg, dict):
                new_msg = dict(msg)
                key = "message" if "message" in new_msg else "content"
                new_msg[key] = replacement["content"]
                rebuilt.append(new_msg)
            else:
                # Mirror extraction: text lives in ``message`` (v1 history
                # objects), falling back to ``content``.
                attr = "message" if hasattr(msg, "message") else "content"
                rewritten = rewrite_message_object(msg, attr, replacement["content"])
                if rewritten is None:
                    return None
                rebuilt.append(rewritten)
        new_kwargs["chat_history"] = rebuilt

    if new_kwargs.get("message") is not None:
        if guard_idx >= len(redacted):
            return None
        new_kwargs["message"] = redacted[guard_idx]["content"]
    elif not chat_history:
        # No known redactable shape was present.
        return None

    return new_kwargs


def _extract_response_text(response: Any) -> str | None:
    try:
        if hasattr(response, "text"):
            text: str | None = response.text
            return text
        if hasattr(response, "message") and hasattr(response.message, "content"):
            parts = response.message.content
            if isinstance(parts, list):
                texts = [p.text for p in parts if hasattr(p, "text")]
                return "\n".join(texts) if texts else None
            return str(parts) if parts else None
        if isinstance(response, dict):
            dict_text: str | None = response.get("text", response.get("message", {}).get("content"))
            return dict_text
    except Exception:
        logger.debug("Failed to extract Cohere response text", exc_info=True)
    return None


# ---------------------------------------------------------------------------
# Apply / revert
# ---------------------------------------------------------------------------


def apply() -> bool:
    global _patched

    if _patched:
        return True

    if importlib.util.find_spec("cohere") is None:
        return False

    import cohere

    patched_any = False

    for cls_name, is_async in [
        ("ClientV2", False),
        ("Client", False),
        ("AsyncClientV2", True),
        ("AsyncClient", True),
    ]:
        try:
            cls = getattr(cohere, cls_name)
            if not hasattr(cls, "chat"):
                continue
            key = f"{cls_name}.chat"
            _originals[key] = cls.chat
            wrap_fn = wrap_async if is_async else wrap_sync
            cls.chat = wrap_fn(
                cls.chat,
                _extract_messages,
                _extract_response_text,
                _apply_redaction,
            )
            patched_any = True
        except (AttributeError, TypeError):
            pass

    _patched = patched_any
    return _patched


def revert() -> None:
    global _patched

    if not _patched:
        return

    try:
        import cohere

        for key, original in _originals.items():
            cls_name, _ = key.split(".")
            try:
                cls = getattr(cohere, cls_name)
                cls.chat = original
            except AttributeError:
                pass
    except ImportError:
        pass

    _originals.clear()
    _patched = False
