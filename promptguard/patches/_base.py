"""
Shared wrapper logic for all SDK patches.

Every patch needs the same pre-call / post-call flow:
1. Extract messages from the SDK's format
2. Call guard.scan() (or scan_async())
3. Handle block/redact/allow based on mode
4. Call the original function
5. Optionally scan the response

This module provides ``wrap_sync`` and ``wrap_async`` so each patch only
needs to supply:
- ``extract_messages(args, kwargs) -> (guard_messages, model, context)``
- ``extract_response(response) -> Optional[str]``
- (optional) ``should_intercept(args, kwargs) -> bool``
"""

import functools
import logging
from collections.abc import Callable
from typing import Any

from promptguard.guard import GuardApiError, PromptGuardBlockedError

logger = logging.getLogger("promptguard")


# -- Shared decision helpers (used by both sync and async wrappers) ----------


def _handle_pre_scan_decision(
    decision: Any,
    get_mode: Callable[[], str],
    apply_redaction: Callable | None,
    args: tuple,
    kwargs: dict,
) -> dict:
    """Process a pre-call guard decision.  Returns (possibly modified) kwargs."""
    if decision is None:
        return kwargs

    if decision.blocked:
        if get_mode() == "enforce":
            raise PromptGuardBlockedError(decision)
        logger.warning(
            "[monitor] PromptGuard would block: %s (event=%s)",
            decision.threat_type,
            decision.event_id,
        )

    if decision.redacted and decision.redacted_messages:
        if get_mode() == "enforce" and apply_redaction:
            return apply_redaction(args, kwargs, decision.redacted_messages)
        logger.warning(
            "[monitor] PromptGuard would redact: %s (event=%s)",
            decision.threat_type,
            decision.event_id,
        )

    return kwargs


def _handle_response_block(resp_decision: Any, get_mode: Callable[[], str]) -> None:
    """Raise if a response scan decision blocks in enforce mode."""
    if resp_decision.blocked and get_mode() == "enforce":
        raise PromptGuardBlockedError(resp_decision)


# -- Public wrappers ---------------------------------------------------------


def wrap_sync(
    original_fn: Callable,
    extract_messages: Callable[..., tuple[list[dict[str, str]], str | None, dict[str, Any]]],
    extract_response: Callable[[Any], str | None],
    apply_redaction: Callable | None = None,
    should_intercept: Callable[..., bool] | None = None,
) -> Callable:
    """Create a sync wrapper that scans before/after the original call.

    Parameters
    ----------
    original_fn:
        The original SDK method (e.g. ``Completions.create``).
    extract_messages:
        ``(args, kwargs) -> (guard_messages, model_name, context_dict)``.
    extract_response:
        ``(response) -> Optional[str]``.  Return ``None`` to skip.
    apply_redaction:
        Optional ``(args, kwargs, redacted_messages) -> kwargs``.
    should_intercept:
        Optional ``(args, kwargs) -> bool``.  When provided, the wrapper
        calls the original function directly if this returns ``False``.
    """

    @functools.wraps(original_fn)
    def wrapper(*args, **kwargs):
        if should_intercept and not should_intercept(args, kwargs):
            return original_fn(*args, **kwargs)

        from promptguard.auto import get_guard_client, get_mode, is_fail_open, should_scan_responses

        guard = get_guard_client()
        if guard is None:
            return original_fn(*args, **kwargs)

        guard_messages, model, context = extract_messages(args, kwargs)

        if guard_messages:
            try:
                decision = guard.scan(
                    messages=guard_messages,
                    direction="input",
                    model=model,
                    context=context,
                )
            except GuardApiError:
                if not is_fail_open():
                    raise
                logger.warning("Guard API unavailable, allowing request (fail_open=True)")
                decision = None

            kwargs = _handle_pre_scan_decision(decision, get_mode, apply_redaction, args, kwargs)

        response = original_fn(*args, **kwargs)

        if should_scan_responses() and response and guard:
            try:
                resp_text = extract_response(response)
                if resp_text:
                    resp_decision = guard.scan(
                        messages=[{"role": "assistant", "content": resp_text}],
                        direction="output",
                        model=model,
                    )
                    _handle_response_block(resp_decision, get_mode)
            except (PromptGuardBlockedError, GuardApiError):
                raise
            except Exception:
                logger.debug("Response scanning failed", exc_info=True)

        return response

    return wrapper


def wrap_async(
    original_fn: Callable,
    extract_messages: Callable[..., tuple[list[dict[str, str]], str | None, dict[str, Any]]],
    extract_response: Callable[[Any], str | None],
    apply_redaction: Callable | None = None,
    should_intercept: Callable[..., bool] | None = None,
) -> Callable:
    """Async variant of ``wrap_sync``.  Same interface."""

    @functools.wraps(original_fn)
    async def wrapper(*args, **kwargs):
        if should_intercept and not should_intercept(args, kwargs):
            return await original_fn(*args, **kwargs)

        from promptguard.auto import get_guard_client, get_mode, is_fail_open, should_scan_responses

        guard = get_guard_client()
        if guard is None:
            return await original_fn(*args, **kwargs)

        guard_messages, model, context = extract_messages(args, kwargs)

        if guard_messages:
            try:
                decision = await guard.scan_async(
                    messages=guard_messages,
                    direction="input",
                    model=model,
                    context=context,
                )
            except GuardApiError:
                if not is_fail_open():
                    raise
                logger.warning("Guard API unavailable, allowing request (fail_open=True)")
                decision = None

            kwargs = _handle_pre_scan_decision(decision, get_mode, apply_redaction, args, kwargs)

        response = await original_fn(*args, **kwargs)

        if should_scan_responses() and response and guard:
            try:
                resp_text = extract_response(response)
                if resp_text:
                    resp_decision = await guard.scan_async(
                        messages=[{"role": "assistant", "content": resp_text}],
                        direction="output",
                        model=model,
                    )
                    _handle_response_block(resp_decision, get_mode)
            except (PromptGuardBlockedError, GuardApiError):
                raise
            except Exception:
                logger.debug("Response scanning failed", exc_info=True)

        return response

    return wrapper
