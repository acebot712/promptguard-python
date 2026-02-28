"""
Google Generative AI SDK patch — wraps
``google.generativeai.GenerativeModel.generate_content`` (sync + async).

Covers direct Google AI usage and frameworks built on top (LangChain's
ChatGoogleGenerativeAI, LlamaIndex's Gemini integration, etc.).
"""

import functools
import importlib.util
import logging
from typing import Any

logger = logging.getLogger("promptguard")

NAME = "google-generativeai"

_original_sync_generate = None
_original_async_generate = None
_patched = False


# ---------------------------------------------------------------------------
# Message extraction (Google-specific)
# ---------------------------------------------------------------------------


def _content_to_guard_format(contents: Any) -> list[dict[str, str]]:
    """Convert google-generativeai content to guard API format."""
    result: list[dict[str, str]] = []

    if isinstance(contents, str):
        result.append({"role": "user", "content": contents})
        return result

    if not isinstance(contents, (list, tuple)):
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


def _extract_response_text(response: Any) -> str | None:
    try:
        if hasattr(response, "text"):
            return response.text
        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            return _extract_text_from_parts(parts)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Wrappers (Google's generate_content takes ``self`` as first arg, so we
# can't directly use the generic wrap_sync/wrap_async which expect a plain
# function.  We use thin wrappers that delegate to the guard client.)
# ---------------------------------------------------------------------------


def _make_sync_wrapper(original_fn):
    @functools.wraps(original_fn)
    def wrapper(self, *args, **kwargs):
        from promptguard.auto import get_guard_client, get_mode, is_fail_open, should_scan_responses
        from promptguard.guard import GuardApiError, PromptGuardBlockedError

        guard = get_guard_client()
        if guard is None:
            return original_fn(self, *args, **kwargs)

        contents = kwargs.get("contents") or (args[0] if args else None)
        model_name = getattr(self, "model_name", None) or getattr(self, "_model_name", None)

        if contents:
            guard_messages = _content_to_guard_format(contents)
            try:
                decision = guard.scan(
                    messages=guard_messages,
                    direction="input",
                    model=str(model_name) if model_name else "gemini",
                    context={"framework": "google-generativeai"},
                )
            except GuardApiError:
                if not is_fail_open():
                    raise
                decision = None

            if decision is not None:
                if decision.blocked:
                    if get_mode() == "enforce":
                        raise PromptGuardBlockedError(decision)
                    logger.warning("[monitor] would block: %s", decision.threat_type)

        response = original_fn(self, *args, **kwargs)

        if should_scan_responses() and response and guard:
            try:
                resp_text = _extract_response_text(response)
                if resp_text:
                    resp_decision = guard.scan(
                        messages=[{"role": "assistant", "content": resp_text}],
                        direction="output",
                        model=str(model_name) if model_name else "gemini",
                    )
                    if resp_decision.blocked and get_mode() == "enforce":
                        raise PromptGuardBlockedError(resp_decision)
            except (PromptGuardBlockedError, GuardApiError):
                raise
            except Exception:
                logger.debug("Response scanning failed", exc_info=True)

        return response

    return wrapper


def _make_async_wrapper(original_fn):
    @functools.wraps(original_fn)
    async def wrapper(self, *args, **kwargs):
        from promptguard.auto import get_guard_client, get_mode, is_fail_open, should_scan_responses
        from promptguard.guard import GuardApiError, PromptGuardBlockedError

        guard = get_guard_client()
        if guard is None:
            return await original_fn(self, *args, **kwargs)

        contents = kwargs.get("contents") or (args[0] if args else None)
        model_name = getattr(self, "model_name", None) or getattr(self, "_model_name", None)

        if contents:
            guard_messages = _content_to_guard_format(contents)
            try:
                decision = await guard.scan_async(
                    messages=guard_messages,
                    direction="input",
                    model=str(model_name) if model_name else "gemini",
                    context={"framework": "google-generativeai"},
                )
            except GuardApiError:
                if not is_fail_open():
                    raise
                decision = None

            if decision is not None:
                if decision.blocked:
                    if get_mode() == "enforce":
                        raise PromptGuardBlockedError(decision)
                    logger.warning("[monitor] would block: %s", decision.threat_type)

        response = await original_fn(self, *args, **kwargs)

        if should_scan_responses() and response and guard:
            try:
                resp_text = _extract_response_text(response)
                if resp_text:
                    resp_decision = await guard.scan_async(
                        messages=[{"role": "assistant", "content": resp_text}],
                        direction="output",
                        model=str(model_name) if model_name else "gemini",
                    )
                    if resp_decision.blocked and get_mode() == "enforce":
                        raise PromptGuardBlockedError(resp_decision)
            except (PromptGuardBlockedError, GuardApiError):
                raise
            except Exception:
                logger.debug("Response scanning failed", exc_info=True)

        return response

    return wrapper


# ---------------------------------------------------------------------------
# Apply / revert
# ---------------------------------------------------------------------------


def apply() -> bool:
    global _original_sync_generate, _original_async_generate, _patched

    if _patched:
        return True

    if importlib.util.find_spec("google.generativeai") is None:
        return False

    try:
        from google.generativeai import GenerativeModel

        _original_sync_generate = GenerativeModel.generate_content
        GenerativeModel.generate_content = _make_sync_wrapper(GenerativeModel.generate_content)
    except (ImportError, AttributeError):
        logger.debug("Could not patch google GenerativeModel.generate_content")

    try:
        from google.generativeai import GenerativeModel

        if hasattr(GenerativeModel, "generate_content_async"):
            _original_async_generate = GenerativeModel.generate_content_async
            GenerativeModel.generate_content_async = _make_async_wrapper(
                GenerativeModel.generate_content_async
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
