"""
AWS Bedrock patch — wraps ``botocore`` Bedrock Runtime ``invoke_model``
and ``converse`` operations.

Covers AWS-native GenAI apps using boto3/botocore to call Bedrock models
(Claude, Titan, Llama, Mistral on Bedrock).

Note: Unlike the other patches, this wraps ``BaseClient._make_api_call``
which has a different call signature (self, operation_name, api_params).
It cannot use the generic ``wrap_sync``/``wrap_async`` from ``_base.py``.
"""

import functools
import importlib.util
import json
import logging
from typing import Any

logger = logging.getLogger("promptguard")

NAME = "boto3-bedrock"

_original_api_call = None
_patched = False


# ---------------------------------------------------------------------------
# Message extraction (Bedrock-specific, handles multiple model formats)
# ---------------------------------------------------------------------------

_BEDROCK_OPERATIONS = frozenset(("InvokeModel", "Converse", "ConverseStream"))


def _extract_messages_from_body(body: Any, model_id: str = "") -> list[dict[str, str]]:
    """Extract messages from a Bedrock request body.

    Bedrock accepts JSON bodies whose schema varies by model provider:
    - Anthropic (Claude): ``messages`` list + optional ``system``
    - Amazon (Titan): ``inputText`` string
    - Meta (Llama): ``prompt`` string
    - Mistral: ``prompt`` string
    - Converse API: ``Messages`` list + optional ``System``
    """
    if isinstance(body, (bytes, bytearray)):
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

    # Anthropic on Bedrock (InvokeModel)
    if "messages" in body:
        _extract_system(body.get("system"), result)
        for msg in body["messages"]:
            if isinstance(msg, dict):
                content = _flatten_content_blocks(msg.get("content", ""))
                result.append({"role": msg.get("role", "user"), "content": content})
        return result

    # Converse API
    if "Messages" in body:
        _extract_system(body.get("System", body.get("system")), result)
        for msg in body["Messages"]:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = _flatten_content_blocks(msg.get("content", ""))
                result.append({"role": role, "content": content})
        return result

    # Amazon Titan
    if "inputText" in body:
        return [{"role": "user", "content": body["inputText"]}]

    # Llama / Mistral
    if "prompt" in body:
        return [{"role": "user", "content": body["prompt"]}]

    return []


def _extract_system(system: Any, result: list[dict[str, str]]) -> None:
    """Extract system prompt from various Bedrock formats into result."""
    if not system:
        return
    if isinstance(system, str):
        result.append({"role": "system", "content": system})
    elif isinstance(system, list):
        texts = [b.get("text", "") for b in system if isinstance(b, dict) and "text" in b]
        if texts:
            result.append({"role": "system", "content": "\n".join(texts)})


def _flatten_content_blocks(content: Any) -> str:
    """Flatten Bedrock content (string or list of content blocks) to text."""
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


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


def _make_api_call_wrapper(original_fn):
    """Wrap botocore's ``_make_api_call`` for Bedrock Runtime operations."""

    @functools.wraps(original_fn)
    def wrapper(self, operation_name, api_params, *args, **kwargs):
        from promptguard.auto import get_guard_client, get_mode, is_fail_open
        from promptguard.guard import GuardApiError, PromptGuardBlockedError

        if operation_name not in _BEDROCK_OPERATIONS:
            return original_fn(self, operation_name, api_params, *args, **kwargs)

        guard = get_guard_client()
        if guard is None:
            return original_fn(self, operation_name, api_params, *args, **kwargs)

        model_id = api_params.get("modelId", api_params.get("ModelId", "bedrock"))

        if operation_name == "InvokeModel":
            guard_messages = _extract_messages_from_body(api_params.get("body", b""), model_id)
        else:
            guard_messages = _extract_messages_from_body(api_params, model_id)

        if guard_messages:
            context = {
                "framework": "aws-bedrock",
                "metadata": {"operation": operation_name, "model_id": model_id},
            }
            try:
                decision = guard.scan(
                    messages=guard_messages,
                    direction="input",
                    model=str(model_id),
                    context=context,
                )
            except GuardApiError:
                if not is_fail_open():
                    raise
                logger.warning("Guard API unavailable, allowing request (fail_open=True)")
                decision = None

            if decision is not None and decision.blocked:
                if get_mode() == "enforce":
                    raise PromptGuardBlockedError(decision)
                logger.warning(
                    "[monitor] PromptGuard would block: %s (event=%s)",
                    decision.threat_type,
                    decision.event_id,
                )

        return original_fn(self, operation_name, api_params, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Apply / revert
# ---------------------------------------------------------------------------


def apply() -> bool:
    """Patch boto3/botocore Bedrock Runtime. Returns True if patched."""
    global _original_api_call, _patched

    if _patched:
        return True

    if importlib.util.find_spec("botocore") is None:
        return False

    import botocore.client

    try:
        _original_api_call = botocore.client.BaseClient._make_api_call
        botocore.client.BaseClient._make_api_call = _make_api_call_wrapper(_original_api_call)
        _patched = True
        return True
    except (ImportError, AttributeError):
        logger.debug("Could not patch botocore BaseClient._make_api_call")
        return False


def revert() -> None:
    """Remove the Bedrock patch."""
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
