"""
OpenAI SDK patch.

Patched call surfaces (sync and async clients):

- ``chat.completions.create()``
- ``chat.completions.parse()`` (when the installed SDK exposes it)
- ``responses.create()`` (when the installed SDK ships the Responses API)

This covers every framework built on top of the OpenAI Python SDK, including
LangChain (ChatOpenAI), CrewAI, AutoGen, Semantic Kernel, and direct usage —
*for calls made through the surfaces above*.  Other OpenAI endpoints
(embeddings, audio, images, batches, fine-tuning) are not scanned.

Responses API scope: the patch extracts scannable text from the basic
``input`` forms — a plain string, or an item list of role/content messages
(string content or ``input_text``/``output_text``/``text`` content parts) —
plus the ``instructions`` param.  Exotic item types (function_call outputs,
reasoning items, etc.) are not scanned.
"""

import importlib.util
import logging
from typing import Any

from promptguard.patches._base import rewrite_message_object, wrap_async, wrap_sync

logger = logging.getLogger("promptguard")

NAME = "openai"

# Maps "Completions.create" / "AsyncResponses.create" / ... -> original method.
_originals: dict[str, Any] = {}
_patched = False


# ---------------------------------------------------------------------------
# Message extraction / redaction (OpenAI-specific)
# ---------------------------------------------------------------------------


def _messages_to_guard_format(messages: Any) -> list[dict[str, str]]:
    """Convert OpenAI-style messages to the guard API format."""
    result: list[dict[str, str]] = []
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
    """Extract messages and model from OpenAI create() call signature.

    ``Completions.create`` is patched as an unbound method and takes
    ``messages`` / ``model`` as keyword-only arguments, so positional args are
    never the payload (``args[0]`` is ``self``).  Read from kwargs only.
    """
    messages = kwargs.get("messages")
    model = kwargs.get("model")
    guard_messages = _messages_to_guard_format(messages) if messages else []
    return guard_messages, str(model) if model else None, {"framework": "openai"}


def _emits_guard_message(msg: Any) -> bool:
    """Whether ``_messages_to_guard_format`` emits a guard message for ``msg``.

    Single source of truth for extraction and redaction so their indices
    stay aligned (skipped entries must not consume a redacted message).
    """
    return isinstance(msg, dict) or (hasattr(msg, "role") and hasattr(msg, "content"))


def _apply_redaction(args, kwargs, redacted: list[dict[str, str]]) -> dict | None:
    """Apply redacted content back into kwargs.

    Mirrors ``_messages_to_guard_format``: only entries that emitted a guard
    message consume a redacted message.  Attribute-based message objects are
    rewritten via a copy; if any message with a redacted counterpart cannot
    be rewritten, returns ``None`` so enforce mode escalates to a block.
    """
    messages = kwargs.get("messages")
    if not messages or not redacted:
        return None
    result: list = []
    guard_idx = 0
    for msg in messages:
        if not _emits_guard_message(msg):
            result.append(msg)
            continue
        replacement = redacted[guard_idx] if guard_idx < len(redacted) else None
        guard_idx += 1
        if replacement is None:
            # No redacted counterpart for a scanned message; forwarding the
            # original would leak flagged content.
            return None
        if isinstance(msg, dict):
            new_msg = dict(msg)
            # Preserve the multimodal shape: array content is rebuilt as a
            # text part rather than collapsed to a bare string.
            if isinstance(new_msg.get("content"), list):
                new_msg["content"] = [{"type": "text", "text": replacement["content"]}]
            else:
                new_msg["content"] = replacement["content"]
            result.append(new_msg)
        else:
            rewritten = rewrite_message_object(msg, "content", replacement["content"])
            if rewritten is None:
                return None
            result.append(rewritten)
    new_kwargs: dict = dict(kwargs)
    new_kwargs["messages"] = result
    return new_kwargs


def _extract_response_content(response: Any) -> str | None:
    """Extract text content from an OpenAI ChatCompletion response."""
    try:
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                content: str | None = choice.message.content
                return content
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                msg_content: str | None = choices[0].get("message", {}).get("content")
                return msg_content
    except Exception:
        logger.debug("Failed to extract OpenAI response text", exc_info=True)
    return None


# ---------------------------------------------------------------------------
# Responses API (client.responses.create)
# ---------------------------------------------------------------------------


def _flatten_responses_content(content: Any) -> str:
    """Flatten Responses API message content (string or content-part list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict):
                if part.get("type") in ("input_text", "output_text", "text") and isinstance(
                    part.get("text"), str
                ):
                    texts.append(part["text"])
            elif getattr(part, "type", None) in (
                "input_text",
                "output_text",
                "text",
            ) and isinstance(getattr(part, "text", None), str):
                texts.append(part.text)
        return "\n".join(texts)
    return str(content) if content is not None else ""


def _is_responses_message_item(item: Any) -> bool:
    """Whether a Responses API input item is a message we extract a guard
    message for.  Single source of truth for extraction and redaction so
    indices stay aligned."""
    if not isinstance(item, dict):
        return False
    if item.get("type") is not None and item.get("type") != "message":
        return False
    return "role" in item or "content" in item


def _responses_input_to_guard_format(kwargs: dict) -> list[dict[str, str]]:
    """Extract guard messages from Responses API params: ``instructions``
    (system), then ``input`` as a plain string or a list of message items."""
    result: list[dict[str, str]] = []

    instructions = kwargs.get("instructions")
    if isinstance(instructions, str) and instructions:
        result.append({"role": "system", "content": instructions})

    input_param = kwargs.get("input")
    if isinstance(input_param, str):
        result.append({"role": "user", "content": input_param})
        return result
    if not isinstance(input_param, list):
        return result

    for item in input_param:
        if not _is_responses_message_item(item):
            continue
        result.append(
            {
                "role": str(item.get("role", "user")),
                "content": _flatten_responses_content(item.get("content")),
            }
        )
    return result


def _extract_responses_messages(
    args, kwargs
) -> tuple[list[dict[str, str]], str | None, dict[str, Any]]:
    model = kwargs.get("model")
    guard_messages = _responses_input_to_guard_format(kwargs)
    return guard_messages, str(model) if model else None, {"framework": "openai"}


def _apply_responses_redaction(args, kwargs, redacted: list[dict[str, str]]) -> dict | None:
    """Map redacted guard messages back onto Responses API kwargs.

    Mirrors ``_responses_input_to_guard_format``: ``instructions`` first
    (when emitted), then string input or message items in order.  Returns
    ``None`` for shapes that cannot be rewritten safely (escalated to a
    block in enforce mode).
    """
    new_kwargs: dict = dict(kwargs)
    guard_idx = 0

    instructions = kwargs.get("instructions")
    if isinstance(instructions, str) and instructions:
        if guard_idx < len(redacted):
            new_kwargs["instructions"] = redacted[guard_idx]["content"]
        guard_idx += 1

    input_param = kwargs.get("input")
    if isinstance(input_param, str):
        if guard_idx >= len(redacted):
            return None
        new_kwargs["input"] = redacted[guard_idx]["content"]
        return new_kwargs

    if not isinstance(input_param, list):
        # Only ``instructions`` was extracted; anything else is unredactable.
        return new_kwargs if guard_idx > 0 else None

    new_input: list = []
    for item in input_param:
        if not _is_responses_message_item(item):
            new_input.append(item)
            continue
        replacement = redacted[guard_idx] if guard_idx < len(redacted) else None
        guard_idx += 1
        if replacement is None:
            return None
        new_item = dict(item)
        # Preserve the structured shape: part lists are rebuilt as a text part.
        if isinstance(new_item.get("content"), list):
            new_item["content"] = [{"type": "input_text", "text": replacement["content"]}]
        else:
            new_item["content"] = replacement["content"]
        new_input.append(new_item)
    new_kwargs["input"] = new_input
    return new_kwargs


def _extract_responses_response_text(response: Any) -> str | None:
    """Extract text from a Responses API response (output_text or items)."""
    try:
        output_text = getattr(response, "output_text", None)
        if output_text is None and isinstance(response, dict):
            output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        output = getattr(response, "output", None)
        if output is None and isinstance(response, dict):
            output = response.get("output")
        if isinstance(output, list):
            texts: list[str] = []
            for item in output:
                if _is_responses_message_item(item):
                    text = _flatten_responses_content(item.get("content"))
                    if text:
                        texts.append(text)
            return "\n".join(texts) if texts else None
    except Exception:
        logger.debug("Failed to extract OpenAI Responses text", exc_info=True)
    return None


# ---------------------------------------------------------------------------
# Apply / revert
# ---------------------------------------------------------------------------


def _patch_method(cls: Any, cls_name: str, method_name: str, *, is_async: bool, config) -> bool:
    """Wrap ``cls.<method_name>`` and record the original.  Returns success."""
    if not hasattr(cls, method_name):
        return False
    key = f"{cls_name}.{method_name}"
    if key in _originals:
        return True
    original = getattr(cls, method_name)
    _originals[key] = original
    wrap_fn = wrap_async if is_async else wrap_sync
    setattr(cls, method_name, wrap_fn(original, *config))
    return True


def _apply_chat_completions_patch() -> bool:
    chat_config = (_extract_messages, _extract_response_content, _apply_redaction)
    patched_any = False
    try:
        from openai.resources.chat.completions import AsyncCompletions, Completions
    except (ImportError, AttributeError):
        logger.debug("Could not patch openai chat completions")
        return False

    for cls, cls_name, is_async in [
        (Completions, "Completions", False),
        (AsyncCompletions, "AsyncCompletions", True),
    ]:
        # ``parse()`` (structured outputs) takes the same messages/model
        # params as ``create()`` — newer SDKs expose it on the same resource.
        for method_name in ("create", "parse"):
            if _patch_method(cls, cls_name, method_name, is_async=is_async, config=chat_config):
                patched_any = True
    return patched_any


def _apply_responses_patch() -> bool:
    responses_config = (
        _extract_responses_messages,
        _extract_responses_response_text,
        _apply_responses_redaction,
    )
    try:
        # Guarded import: older ``openai`` versions have no Responses API —
        # the chat.completions patch still applies without it.
        from openai.resources.responses import AsyncResponses, Responses
    except (ImportError, AttributeError):
        logger.debug("openai Responses API not available; skipping responses patch")
        return False

    patched_any = False
    for cls, cls_name, is_async in [
        (Responses, "Responses", False),
        (AsyncResponses, "AsyncResponses", True),
    ]:
        if _patch_method(cls, cls_name, "create", is_async=is_async, config=responses_config):
            patched_any = True
    return patched_any


def apply() -> bool:
    """Patch the OpenAI SDK. Returns True if successfully patched."""
    global _patched

    if _patched:
        return True

    if importlib.util.find_spec("openai") is None:
        return False

    chat_patched = _apply_chat_completions_patch()
    responses_patched = _apply_responses_patch()

    _patched = chat_patched or responses_patched
    if _patched:
        logger.debug(
            "openai patch active (chat.completions=%s, responses=%s)",
            chat_patched,
            responses_patched,
        )
    return _patched


def revert() -> None:
    """Remove the OpenAI SDK patch."""
    global _patched

    if not _patched:
        return

    try:
        from openai.resources.chat.completions import AsyncCompletions, Completions

        classes = {"Completions": Completions, "AsyncCompletions": AsyncCompletions}
    except (ImportError, AttributeError):
        classes = {}

    try:
        from openai.resources.responses import AsyncResponses, Responses

        classes["Responses"] = Responses
        classes["AsyncResponses"] = AsyncResponses
    except (ImportError, AttributeError):
        pass

    for key, original in _originals.items():
        cls_name, method_name = key.split(".")
        cls = classes.get(cls_name)
        if cls is not None:
            setattr(cls, method_name, original)

    _originals.clear()
    _patched = False
