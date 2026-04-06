"""
Google Generative AI SDK patch - wraps
``google.generativeai.GenerativeModel.generate_content`` (sync + async).

Covers direct Google AI usage and frameworks built on top (LangChain's
ChatGoogleGenerativeAI, LlamaIndex's Gemini integration, etc.).
"""

import importlib.util
import logging
from typing import Any

from promptguard.patches._base import wrap_async, wrap_sync

logger = logging.getLogger("promptguard")

NAME = "google-generativeai"

_original_sync_generate = None
_original_async_generate = None
_patched = False


# -- Message extraction (Google-specific) ------------------------------------


def _content_to_guard_format(contents: Any) -> list[dict[str, str]]:
    """Convert google-generativeai content to guard API format."""
    result: list[dict[str, str]] = []

    if isinstance(contents, str):
        result.append({"role": "user", "content": contents})
        return result

    if not isinstance(contents, list | tuple):
        result.append({"role": "user", "content": str(contents)})
        return result

    for item in contents:
        if isinstance(item, str):
            result.append({"role": "user", "content": item})
        elif isinstance(item, dict):
            role = item.get("role", "user")
            parts = item.get("parts", [])
            text = _extract_text_from_parts(parts)
            result.append({"role": role, "content": text})
        elif hasattr(item, "role") and hasattr(item, "parts"):
            text = _extract_text_from_parts(item.parts)
            result.append({"role": item.role or "user", "content": text})
        else:
            result.append({"role": "user", "content": str(item)})

    return result


def _extract_text_from_parts(parts: Any) -> str:
    if isinstance(parts, str):
        return parts
    if not parts:
        return ""
    texts = []
    for part in parts:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and "text" in part:
            texts.append(part["text"])
        elif hasattr(part, "text"):
            texts.append(part.text)
    return "\n".join(texts)


# -- Extractors compatible with _base.wrap_sync / wrap_async -----------------


def _extract_messages(
    args: tuple, kwargs: dict
) -> tuple[list[dict[str, str]], str | None, dict[str, Any]]:
    """Extract guard messages from GenerativeModel.generate_content args.

    ``args[0]`` is ``self`` (the GenerativeModel instance) because we're
    patching an unbound method on the class.
    """
    model_instance = args[0] if args else None
    contents = kwargs.get("contents") or (args[1] if len(args) > 1 else None)
    model_name = getattr(model_instance, "model_name", None) or getattr(
        model_instance, "_model_name", None
    )
    guard_messages = _content_to_guard_format(contents) if contents else []
    return (
        guard_messages,
        str(model_name) if model_name else "gemini",
        {"framework": "google-generativeai"},
    )


def _extract_response_text(response: Any) -> str | None:
    try:
        if hasattr(response, "text"):
            return response.text
        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            return _extract_text_from_parts(parts)
    except Exception:
        logger.debug("Failed to extract Google response text", exc_info=True)
    return None


# -- Apply / revert ----------------------------------------------------------


def apply() -> bool:
    global _original_sync_generate, _original_async_generate, _patched

    if _patched:
        return True

    if importlib.util.find_spec("google.generativeai") is None:
        return False

    try:
        from google.generativeai import GenerativeModel

        _original_sync_generate = GenerativeModel.generate_content
        GenerativeModel.generate_content = wrap_sync(
            GenerativeModel.generate_content,
            _extract_messages,
            _extract_response_text,
        )
    except (ImportError, AttributeError):
        logger.debug("Could not patch google GenerativeModel.generate_content")

    try:
        from google.generativeai import GenerativeModel

        if hasattr(GenerativeModel, "generate_content_async"):
            _original_async_generate = GenerativeModel.generate_content_async
            GenerativeModel.generate_content_async = wrap_async(
                GenerativeModel.generate_content_async,
                _extract_messages,
                _extract_response_text,
            )
    except (ImportError, AttributeError):
        logger.debug("Could not patch google GenerativeModel.generate_content_async")

    _patched = _original_sync_generate is not None
    return _patched


def revert() -> None:
    global _original_sync_generate, _original_async_generate, _patched

    if not _patched:
        return

    try:
        from google.generativeai import GenerativeModel

        if _original_sync_generate:
            GenerativeModel.generate_content = _original_sync_generate
        if _original_async_generate:
            GenerativeModel.generate_content_async = _original_async_generate
    except (ImportError, AttributeError):
        pass

    _original_sync_generate = None
    _original_async_generate = None
    _patched = False
