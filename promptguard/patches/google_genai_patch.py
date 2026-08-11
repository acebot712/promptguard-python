"""
Google Gen AI SDK patch — wraps ``google.genai.models.Models.generate_content``
and its streaming and async siblings.

This is the SDK Google actually ships today. ``google_patch.py`` covers
``google-generativeai``, which Google deprecated when Gemini 2.0 landed (its own
PyPI page opens with "[Deprecated]"). Both are patched: existing customers are
still on the old one, and dropping it to chase the new one would be the same
silent-unprotection bug in the other direction.

Two things this patch can do that the legacy one cannot, both because the new
signature is ``generate_content(self, *, model, contents, config=None)`` —
keyword-only, where the old one took ``contents`` positionally:

* **Redaction works.** ``_base.wrap_sync``'s redaction contract rewrites kwargs,
  so a ``redact`` decision here rewrites the prompt instead of escalating to a
  block the way ``google_patch`` has to.
* **The system prompt is scanned.** It lives at ``config.system_instruction``,
  which is the highest-authority text in the request and is not part of
  ``contents`` at all.
"""

import importlib.util
import json
import logging
from typing import Any

from promptguard.patches._base import wrap_async, wrap_sync

logger = logging.getLogger("promptguard")

NAME = "google-genai"

# Import names that mean "this patch is relevant". `auto.py` uses these to tell
# "the customer does not use Gemini" apart from "the customer uses Gemini and we
# failed to hook it" — two situations that used to look identical.
DETECTS = ("google.genai",)

_originals: dict[tuple[str, str], Any] = {}
_patched = False


# -- Message extraction ------------------------------------------------------


def _part_text(part: Any) -> str:
    """Everything scannable in one Part, dict-shaped or model-shaped.

    Mirrors ``GeminiProvider._part_text`` on the platform side. Tool-call
    arguments are read deliberately: exfiltration through a tool call is written
    into the arguments, not into the visible prose.
    """
    found: list[str] = []

    def _get(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    if isinstance(part, str):
        return part

    text = _get(part, "text")
    if isinstance(text, str) and text:
        found.append(text)

    for call_key in ("function_call", "functionCall"):
        call = _get(part, call_key)
        if call is None:
            continue
        name = _get(call, "name")
        if isinstance(name, str) and name:
            found.append(name)
        args = _get(call, "args")
        if isinstance(args, dict) and args:
            found.append(json.dumps(args, sort_keys=True, default=str))
        elif isinstance(args, str) and args:
            found.append(args)

    for resp_key in ("function_response", "functionResponse"):
        resp = _get(part, resp_key)
        if resp is None:
            continue
        payload = _get(resp, "response")
        if isinstance(payload, dict) and payload:
            found.append(json.dumps(payload, sort_keys=True, default=str))
        elif isinstance(payload, str) and payload:
            found.append(payload)

    for code_key in ("executable_code", "executableCode"):
        code = _get(part, code_key)
        if code is not None:
            body = _get(code, "code")
            if isinstance(body, str) and body:
                found.append(body)

    for result_key in ("code_execution_result", "codeExecutionResult"):
        result = _get(part, result_key)
        if result is not None:
            out = _get(result, "output")
            if isinstance(out, str) and out:
                found.append(out)

    return "\n".join(found)


def _contents_to_guard_format(contents: Any) -> list[dict[str, str]]:
    """Flatten ``contents`` into guard messages.

    ``contents`` is deliberately permissive in this SDK: a bare string, a Part,
    a Content, or a list mixing all of them. Every one of those forms is handled
    rather than assumed away, because the form a customer happens to use is not
    something we get to choose.
    """
    if contents is None:
        return []
    if isinstance(contents, str):
        return [{"role": "user", "content": contents}]
    if not isinstance(contents, list | tuple):
        contents = [contents]

    result: list[dict[str, str]] = []
    for item in contents:
        if isinstance(item, str):
            result.append({"role": "user", "content": item})
            continue

        role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
        parts = item.get("parts") if isinstance(item, dict) else getattr(item, "parts", None)

        if parts is not None:
            text = "\n".join(t for t in (_part_text(p) for p in parts) if t)
        else:
            # A bare Part handed in where a Content was expected.
            text = _part_text(item)

        if text:
            result.append({"role": str(role or "user"), "content": text})
    return result


def _system_instruction_text(config: Any) -> str:
    """The system prompt, wherever this SDK's config object keeps it."""
    if config is None:
        return ""
    instruction = (
        config.get("system_instruction")
        if isinstance(config, dict)
        else getattr(config, "system_instruction", None)
    )
    if instruction is None:
        return ""
    if isinstance(instruction, str):
        return instruction
    parts = (
        instruction.get("parts")
        if isinstance(instruction, dict)
        else getattr(instruction, "parts", None)
    )
    if parts is not None:
        return "\n".join(t for t in (_part_text(p) for p in parts) if t)
    return _part_text(instruction)


def _extract_messages(
    args: tuple, kwargs: dict
) -> tuple[list[dict[str, str]], str | None, dict[str, Any]]:
    """``(args, kwargs) -> (guard_messages, model, context)``.

    ``args[0]`` is ``self`` (the ``Models`` instance); everything else is
    keyword-only in this SDK, so there is no positional fallback to guess at.
    """
    messages: list[dict[str, str]] = []

    system_text = _system_instruction_text(kwargs.get("config"))
    if system_text:
        messages.append({"role": "system", "content": system_text})

    messages.extend(_contents_to_guard_format(kwargs.get("contents")))

    model = kwargs.get("model")
    return messages, str(model) if model else "gemini", {"framework": NAME}


def _extract_response_text(response: Any) -> str | None:
    try:
        candidates = getattr(response, "candidates", None)
        if candidates:
            texts: list[str] = []
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                for part in getattr(content, "parts", None) or ():
                    part_text = _part_text(part)
                    if part_text:
                        texts.append(part_text)
            if texts:
                return "\n".join(texts)
        text = getattr(response, "text", None)
        return text if isinstance(text, str) else None
    except Exception:
        logger.debug("Failed to extract google-genai response text", exc_info=True)
        return None


def _apply_redaction(_args: tuple, kwargs: dict, redacted: list[dict[str, str]]) -> dict | None:
    """Rewrite ``contents`` with the redacted text.

    Only the non-system messages map back onto ``contents``; the system
    instruction lives in ``config`` and is left alone, so a redaction that only
    fires on the system prompt returns None and fails safe to a block rather
    than silently forwarding it.
    """
    body = [m for m in redacted if m.get("role") != "system"]
    if not body:
        return None
    new_kwargs = dict(kwargs)
    new_kwargs["contents"] = [
        {"role": m.get("role", "user"), "parts": [{"text": m.get("content", "")}]} for m in body
    ]
    return new_kwargs


# -- Apply / revert ----------------------------------------------------------

# (class attribute, is_async) for every surface we wrap. Streaming is included
# because a streamed generation is still a generation: leaving it out would mean
# the scanner silently stops applying the moment a customer passes stream=True,
# which is the failure this whole patch exists to remove.
_SURFACES: tuple[tuple[str, str, bool], ...] = (
    ("Models", "generate_content", False),
    ("Models", "generate_content_stream", False),
    ("AsyncModels", "generate_content", True),
    ("AsyncModels", "generate_content_stream", True),
)


def apply() -> bool:
    global _patched

    if _patched:
        return True
    if importlib.util.find_spec("google.genai") is None:
        return False

    try:
        from google.genai import models as genai_models
    except ImportError:
        logger.debug("google.genai present but google.genai.models not importable")
        return False

    for class_name, method_name, is_async in _SURFACES:
        klass = getattr(genai_models, class_name, None)
        if klass is None:
            continue
        original = getattr(klass, method_name, None)
        if original is None:
            continue
        try:
            wrap = wrap_async if is_async else wrap_sync
            setattr(
                klass,
                method_name,
                wrap(original, _extract_messages, _extract_response_text, _apply_redaction),
            )
            _originals[(class_name, method_name)] = original
        except (AttributeError, TypeError):
            logger.debug("Could not patch google.genai %s.%s", class_name, method_name)

    _patched = bool(_originals)
    return _patched


def revert() -> None:
    global _patched

    if not _patched:
        return
    try:
        from google.genai import models as genai_models

        for (class_name, method_name), original in _originals.items():
            klass = getattr(genai_models, class_name, None)
            if klass is not None:
                setattr(klass, method_name, original)
    except ImportError:
        pass

    _originals.clear()
    _patched = False
