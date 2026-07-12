"""
AWS Bedrock patch - wraps ``botocore`` Bedrock Runtime ``invoke_model``
and ``converse`` operations.

Covers AWS-native GenAI apps using boto3/botocore to call Bedrock models
(Claude, Titan, Llama, Mistral on Bedrock).
"""

import importlib.util
import json
import logging
from typing import Any

from promptguard.patches._base import wrap_sync

logger = logging.getLogger("promptguard")

NAME = "boto3-bedrock"

_original_api_call = None
_patched = False

_BEDROCK_OPERATIONS = frozenset(("InvokeModel", "Converse", "ConverseStream"))


# -- Intercept filter --------------------------------------------------------


def _should_intercept(args: tuple, kwargs: dict) -> bool:
    """Only intercept Bedrock Runtime operations, not all botocore calls."""
    operation_name = args[1] if len(args) > 1 else kwargs.get("operation_name")
    return operation_name in _BEDROCK_OPERATIONS


# -- Message extraction (Bedrock-specific, handles multiple model formats) ---


def _extract_guard_messages(
    args: tuple, kwargs: dict
) -> tuple[list[dict[str, str]], str | None, dict[str, Any]]:
    """Extract guard messages from _make_api_call(self, operation_name, api_params)."""
    if len(args) < 3:
        return [], None, {}

    operation_name = args[1]
    api_params = args[2]
    model_id = api_params.get("modelId", api_params.get("ModelId", "bedrock"))

    if operation_name == "InvokeModel":
        guard_messages = _extract_messages_from_body(api_params.get("body", b""), model_id)
    else:
        guard_messages = _extract_messages_from_body(api_params, model_id)

    context: dict[str, Any] = {
        "framework": "aws-bedrock",
        "metadata": {"operation": operation_name, "model_id": model_id},
    }
    return guard_messages, str(model_id), context


def _extract_messages_from_body(body: Any, model_id: str = "") -> list[dict[str, str]]:
    """Extract messages from a Bedrock request body.

    Bedrock accepts JSON bodies whose schema varies by model provider:
    - Anthropic (Claude): ``messages`` list + optional ``system``
    - Amazon (Titan): ``inputText`` string
    - Meta (Llama): ``prompt`` string
    - Mistral: ``prompt`` string
    - Converse API: ``Messages`` list + optional ``System``
    """
    if isinstance(body, bytes | bytearray):
        try:
            body = json.loads(body)
        except (ValueError, TypeError):  # includes JSONDecodeError + UnicodeDecodeError
            return []
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except (ValueError, TypeError):  # includes JSONDecodeError + UnicodeDecodeError
            return [{"role": "user", "content": body}]
    if not isinstance(body, dict):
        return []

    result: list[dict[str, str]] = []

    if "messages" in body:
        _extract_system(body.get("system"), result)
        for msg in body["messages"]:
            if _emits_guard_message(msg):
                content = _flatten_content_blocks(msg.get("content", ""))
                result.append({"role": msg.get("role", "user"), "content": content})
        return result

    if "Messages" in body:
        _extract_system(body.get("System", body.get("system")), result)
        for msg in body["Messages"]:
            if _emits_guard_message(msg):
                role = msg.get("role", "user")
                content = _flatten_content_blocks(msg.get("content", ""))
                result.append({"role": role, "content": content})
        return result

    if "inputText" in body:
        return [{"role": "user", "content": body["inputText"]}]

    if "prompt" in body:
        return [{"role": "user", "content": body["prompt"]}]

    return []


def _emits_guard_message(msg: Any) -> bool:
    """Whether ``_extract_messages_from_body`` emits a guard message for a
    ``messages`` / ``Messages`` entry.

    Single source of truth shared by extraction and redaction: entries that
    are skipped during extraction (non-dicts) must not consume a redacted
    message during redaction, otherwise the indices drift and the ORIGINAL
    flagged content of a later entry is forwarded unredacted.
    """
    return isinstance(msg, dict)


def _system_produces_message(system: Any) -> bool:
    """Whether ``_extract_system`` would emit a guard message for ``system``.

    Kept in lockstep with ``_extract_system`` so redaction offsets line up.
    """
    if not system:
        return False
    if isinstance(system, str):
        return True
    if isinstance(system, list):
        return any(isinstance(b, dict) and "text" in b for b in system)
    return False


def _apply_redaction(args: tuple, kwargs: dict, redacted: list[dict[str, str]]) -> dict | None:
    """Write redacted content back into the Bedrock request.

    ``api_params`` is passed positionally (``_make_api_call(self, op, params)``)
    so we mutate it in place — that same dict object is what the wrapped
    original call receives.  Returns ``kwargs`` unchanged on success, or
    ``None`` when the request shape cannot be rewritten so ``_base``
    escalates the redact decision to a block instead of forwarding the
    original (unredacted) content.
    """
    if len(args) < 3 or not redacted:
        return None

    operation_name = args[1]
    api_params = args[2]
    if not isinstance(api_params, dict):
        return None

    if operation_name == "InvokeModel":
        applied = _redact_invoke_body(api_params, redacted)
    else:
        applied = _redact_converse_params(api_params, redacted)
    return kwargs if applied else None


def _redact_invoke_body(api_params: dict, redacted: list[dict[str, str]]) -> bool:
    """Redact an InvokeModel body in place.  Returns False when it cannot."""
    raw = api_params.get("body", b"")
    was_bytes = isinstance(raw, bytes | bytearray)
    body: Any = raw
    if was_bytes:
        try:
            body = json.loads(raw)
        except (ValueError, TypeError):  # includes JSONDecodeError + UnicodeDecodeError
            return False
    elif isinstance(raw, str):
        try:
            body = json.loads(raw)
        except (ValueError, TypeError):  # includes JSONDecodeError + UnicodeDecodeError
            # A bare string prompt maps to a single guard message.
            api_params["body"] = redacted[0]["content"]
            return True
    if not isinstance(body, dict):
        return False

    if not _redact_body_dict(body, redacted, system_key="system"):
        return False
    api_params["body"] = json.dumps(body).encode() if was_bytes else json.dumps(body)
    return True


def _redact_converse_params(api_params: dict, redacted: list[dict[str, str]]) -> bool:
    # Converse passes messages/system directly on api_params.  Real botocore
    # uses lowercase keys (``messages`` / ``system``); older/hand-rolled callers
    # sometimes use the capitalized forms, so prefer those only when present.
    system_key = "System" if "System" in api_params else "system"
    messages_key = "Messages" if "Messages" in api_params else "messages"
    return _redact_body_dict(
        api_params,
        redacted,
        system_key=system_key,
        messages_key=messages_key,
        block_shaped=True,
    )


def _redact_body_dict(
    body: dict,
    redacted: list[dict[str, str]],
    system_key: str,
    messages_key: str = "messages",
    block_shaped: bool = False,
) -> bool:
    """Redact a parsed request body in place, mirroring extraction order.

    Returns False when the body has no shape we know how to rewrite, or when
    an entry that emitted a guard message has no redacted counterpart —
    forwarding it would leak the exact content the guard flagged.
    """

    # Converse requires content/system as block lists (``[{"text": ...}]``);
    # InvokeModel provider bodies (Anthropic/Titan/Llama) take plain strings.
    def _fmt(content: str) -> Any:
        return [{"text": content}] if block_shaped else content

    if messages_key in body:
        messages = body.get(messages_key)
        if not isinstance(messages, list):
            return False
        # Mirror extraction: the system prompt (when it produced a guard
        # message) is guard index 0, then each emitting entry consumes the
        # next redacted message.  Non-emitting entries were never scanned and
        # must not advance the guard index.
        guard_idx = 0
        if _system_produces_message(body.get(system_key)):
            body[system_key] = _fmt(redacted[0]["content"])
            guard_idx = 1
        for msg in messages:
            if not _emits_guard_message(msg):
                continue
            if guard_idx >= len(redacted):
                # A scanned entry without a redacted counterpart cannot be
                # rewritten; escalate instead of forwarding it unredacted.
                return False
            msg["content"] = _fmt(redacted[guard_idx]["content"])
            guard_idx += 1
        return True

    # Single-field prompt formats (Titan inputText / Llama/Mistral prompt).
    # Extraction emits exactly one guard message for these (and ignores any
    # ``system`` field), so the redacted counterpart is always index 0.
    if "inputText" in body:
        body["inputText"] = redacted[0]["content"]
        return True
    if "prompt" in body:
        body["prompt"] = redacted[0]["content"]
        return True

    return False


def _extract_system(system: Any, result: list[dict[str, str]]) -> None:
    if not system:
        return
    if isinstance(system, str):
        result.append({"role": "system", "content": system})
    elif isinstance(system, list):
        texts = [b.get("text", "") for b in system if isinstance(b, dict) and "text" in b]
        if texts:
            result.append({"role": "system", "content": "\n".join(texts)})


def _flatten_content_blocks(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif "text" in block:
                    texts.append(block["text"])
        return "\n".join(texts)
    return str(content)


# -- Response extraction (for scan_responses=True) ---------------------------


def _extract_response(response: Any) -> str | None:
    """Extract assistant text from a Bedrock response for output scanning.

    Handles both surfaces:
    - Converse / ConverseStream: text lives in ``output.message.content[].text``.
    - InvokeModel: the model output is a JSON body whose schema varies by
      provider (Anthropic / Titan / Llama / Mistral), mirroring the request
      body parser above.

    Returns ``None`` for any shape we don't recognise so the caller simply
    skips output scanning rather than scanning garbage.
    """
    if not isinstance(response, dict):
        return None

    # Converse: {"output": {"message": {"role": ..., "content": [{"text": ...}]}}}
    output = response.get("output")
    if isinstance(output, dict):
        message = output.get("message")
        if isinstance(message, dict):
            text = _flatten_content_blocks(message.get("content", ""))
            return text or None

    # InvokeModel: the provider body is a StreamingBody / bytes / str.
    if "body" in response:
        return _extract_invoke_response_text(response)

    return None


def _read_response_body(response: dict) -> bytes | bytearray | str | None:
    """Read the InvokeModel response body without stealing it from the caller.

    A botocore ``StreamingBody`` can only be read once, so after we consume it
    we replace ``response["body"]`` with a fresh stream over the same bytes so
    the application's own ``response["body"].read()`` still works.
    """
    raw = response.get("body")
    if isinstance(raw, bytes | bytearray | str):
        return raw

    read = getattr(raw, "read", None)
    if read is None:
        return None
    try:
        data = read()
    except Exception:
        logger.debug("Failed to read Bedrock InvokeModel response body", exc_info=True)
        return None

    if not isinstance(data, bytes | bytearray | str):
        return None

    if isinstance(data, bytes | bytearray):
        try:
            import io

            from botocore.response import StreamingBody

            response["body"] = StreamingBody(io.BytesIO(data), len(data))
        except Exception:
            # If we can't rebuild a StreamingBody, hand the raw bytes back so the
            # caller at least still has the payload.
            response["body"] = data
    return data


def _extract_invoke_response_text(response: dict) -> str | None:
    data = _read_response_body(response)
    if data is None:
        return None
    try:
        parsed = json.loads(data)
    except (ValueError, TypeError):  # includes JSONDecodeError + UnicodeDecodeError
        return None
    if not isinstance(parsed, dict):
        return None

    # Anthropic messages API: {"content": [{"type": "text", "text": ...}], ...}
    content = parsed.get("content")
    if isinstance(content, list):
        text = _flatten_content_blocks(content)
        if text:
            return text

    # Anthropic legacy text-completion: {"completion": "..."}
    if isinstance(parsed.get("completion"), str):
        return parsed["completion"] or None

    # Amazon Titan: {"results": [{"outputText": "..."}]}
    results = parsed.get("results")
    if isinstance(results, list):
        joined = "\n".join(
            r["outputText"] for r in results if isinstance(r, dict) and r.get("outputText")
        )
        if joined:
            return joined

    # Meta Llama: {"generation": "..."}
    if isinstance(parsed.get("generation"), str):
        return parsed["generation"] or None

    # Mistral: {"outputs": [{"text": "..."}]}
    outputs = parsed.get("outputs")
    if isinstance(outputs, list):
        joined = "\n".join(o["text"] for o in outputs if isinstance(o, dict) and o.get("text"))
        if joined:
            return joined

    return None


# -- Apply / revert ----------------------------------------------------------


def apply() -> bool:
    global _original_api_call, _patched

    if _patched:
        return True

    if importlib.util.find_spec("botocore") is None:
        return False

    import botocore.client

    try:
        _original_api_call = botocore.client.BaseClient._make_api_call
        botocore.client.BaseClient._make_api_call = wrap_sync(
            _original_api_call,
            _extract_guard_messages,
            _extract_response,
            apply_redaction=_apply_redaction,
            should_intercept=_should_intercept,
        )
        _patched = True
        return True
    except (ImportError, AttributeError):
        logger.debug("Could not patch botocore BaseClient._make_api_call")
        return False


def revert() -> None:
    global _original_api_call, _patched

    if not _patched:
        return

    try:
        import botocore.client

        if _original_api_call:
            botocore.client.BaseClient._make_api_call = _original_api_call
    except (ImportError, AttributeError):
        pass

    _original_api_call = None
    _patched = False
