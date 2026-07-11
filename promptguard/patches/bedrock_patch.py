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
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return [{"role": "user", "content": body}]
    if not isinstance(body, dict):
        return []

    result: list[dict[str, str]] = []

    if "messages" in body:
        _extract_system(body.get("system"), result)
        for msg in body["messages"]:
            if isinstance(msg, dict):
                content = _flatten_content_blocks(msg.get("content", ""))
                result.append({"role": msg.get("role", "user"), "content": content})
        return result

    if "Messages" in body:
        _extract_system(body.get("System", body.get("system")), result)
        for msg in body["Messages"]:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = _flatten_content_blocks(msg.get("content", ""))
                result.append({"role": role, "content": content})
        return result

    if "inputText" in body:
        return [{"role": "user", "content": body["inputText"]}]

    if "prompt" in body:
        return [{"role": "user", "content": body["prompt"]}]

    return []


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


def _apply_redaction(args: tuple, kwargs: dict, redacted: list[dict[str, str]]) -> dict:
    """Write redacted content back into the Bedrock request.

    ``api_params`` is passed positionally (``_make_api_call(self, op, params)``)
    so we mutate it in place — that same dict object is what the wrapped
    original call receives.  Returns ``kwargs`` unchanged.
    """
    if len(args) < 3 or not redacted:
        return kwargs

    operation_name = args[1]
    api_params = args[2]
    if not isinstance(api_params, dict):
        return kwargs

    if operation_name == "InvokeModel":
        _redact_invoke_body(api_params, redacted)
    else:
        _redact_converse_params(api_params, redacted)
    return kwargs


def _redact_invoke_body(api_params: dict, redacted: list[dict[str, str]]) -> None:
    raw = api_params.get("body", b"")
    was_bytes = isinstance(raw, bytes | bytearray)
    body: Any = raw
    if was_bytes:
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
    elif isinstance(raw, str):
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # A bare string prompt maps to a single guard message.
            if redacted:
                api_params["body"] = redacted[0]["content"]
            return
    if not isinstance(body, dict):
        return

    _redact_body_dict(body, redacted, system_key="system")
    api_params["body"] = json.dumps(body).encode() if was_bytes else json.dumps(body)


def _redact_converse_params(api_params: dict, redacted: list[dict[str, str]]) -> None:
    # Converse passes Messages/System directly on api_params.
    system_key = "System" if "System" in api_params else "system"
    _redact_body_dict(api_params, redacted, system_key=system_key, messages_key="Messages")


def _redact_body_dict(
    body: dict,
    redacted: list[dict[str, str]],
    system_key: str,
    messages_key: str = "messages",
) -> None:
    offset = 0
    system = body.get(system_key)
    if _system_produces_message(system) and redacted:
        body[system_key] = redacted[0]["content"]
        offset = 1

    messages = body.get(messages_key)
    if isinstance(messages, list):
        for i, msg in enumerate(messages):
            idx = i + offset
            if idx < len(redacted) and isinstance(msg, dict):
                msg["content"] = redacted[idx]["content"]
        return

    # Single-field prompt formats (Titan inputText / Llama/Mistral prompt).
    if "inputText" in body and redacted:
        body["inputText"] = (
            redacted[offset]["content"] if offset < len(redacted) else body["inputText"]
        )
    elif "prompt" in body and redacted:
        body["prompt"] = redacted[offset]["content"] if offset < len(redacted) else body["prompt"]


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
            lambda _: None,
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
