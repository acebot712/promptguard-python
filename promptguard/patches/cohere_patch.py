"""
Cohere SDK patch — wraps ``cohere.Client.chat`` / ``ClientV2.chat``
(sync + async).

Covers direct Cohere usage and frameworks that use it (Haystack, LangChain).
"""

import importlib.util
import logging
from typing import Any

from promptguard.patches._base import wrap_async, wrap_sync

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


def _extract_response_text(response: Any) -> str | None:
    try:
        if hasattr(response, "text"):
            return response.text
        if hasattr(response, "message") and hasattr(response.message, "content"):
            parts = response.message.content
            if isinstance(parts, list):
                texts = [p.text for p in parts if hasattr(p, "text")]
                return "\n".join(texts) if texts else None
            return str(parts) if parts else None
        if isinstance(response, dict):
            return response.get("text", response.get("message", {}).get("content"))
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
            cls.chat = wrap_fn(cls.chat, _extract_messages, _extract_response_text)
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
