"""
Shared wrapper logic for all SDK patches.

Every patch needs the same pre-call / post-call flow:
1. Extract messages from the SDK's format
2. Call guard.scan() (or scan_async())
3. Handle block/redact/allow based on mode
4. Call the original function
5. Optionally scan the response

This module provides ``wrap_sync`` and ``wrap_async`` so each patch only
needs to supply two functions:
- ``extract_messages(args, kwargs) -> (guard_messages, model, context)``
- ``extract_response(response) -> Optional[str]``
"""

import functools
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("promptguard")


def wrap_sync(
    original_fn: Callable,
    extract_messages: Callable[..., tuple[list[dict[str, str]], str | None, dict[str, Any]]],
    extract_response: Callable[[Any], str | None],
    apply_redaction: Callable | None = None,
) -> Callable:
    """Create a sync wrapper that scans before/after the original call.

    Parameters
    ----------
    original_fn:
        The original SDK method (e.g. ``Completions.create``).
    extract_messages:
        ``(args, kwargs) -> (guard_messages, model_name, context_dict)``.
        Returns the messages to scan, the model name, and framework context.
    extract_response:
        ``(response) -> Optional[str]``.  Extracts text from the response
        for output scanning.  Return ``None`` to skip.
    apply_redaction:
        Optional ``(args, kwargs, redacted_messages) -> kwargs`` that
        applies redacted content back into the call's kwargs.
    """

    @functools.wraps(original_fn)
    def wrapper(*args, **kwargs):
        from promptguard.auto import get_guard_client, get_mode, is_fail_open, should_scan_responses
        from promptguard.guard import GuardApiError, PromptGuardBlockedError

        guard = get_guard_client()
        if guard is None:
            return original_fn(*args, **kwargs)

        # -- Pre-call scan ---------------------------------------------------
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

            if decision is not None:
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
                        kwargs = apply_redaction(args, kwargs, decision.redacted_messages)
                    else:
                        logger.warning(
                            "[monitor] PromptGuard would redact: %s (event=%s)",
                            decision.threat_type,
                            decision.event_id,
                        )

        # -- Original call ---------------------------------------------------
        response = original_fn(*args, **kwargs)

        # -- Post-call scan --------------------------------------------------
        if should_scan_responses() and response and guard:
            try:
                resp_text = extract_response(response)
                if resp_text:
                    resp_decision = guard.scan(
                        messages=[{"role": "assistant", "content": resp_text}],
                        direction="output",
                        model=model,
                    )
                    if resp_decision.blocked and get_mode() == "enforce":
                        raise PromptGuardBlockedError(resp_decision)
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
) -> Callable:
    """Async variant of ``wrap_sync``.  Same interface."""

    @functools.wraps(original_fn)
    async def wrapper(*args, **kwargs):
        from promptguard.auto import get_guard_client, get_mode, is_fail_open, should_scan_responses
        from promptguard.guard import GuardApiError, PromptGuardBlockedError

        guard = get_guard_client()
        if guard is None:
            return await original_fn(*args, **kwargs)

        # -- Pre-call scan ---------------------------------------------------
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

            if decision is not None:
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
                        kwargs = apply_redaction(args, kwargs, decision.redacted_messages)
                    else:
                        logger.warning(
                            "[monitor] PromptGuard would redact: %s (event=%s)",
                            decision.threat_type,
                            decision.event_id,
                        )

        # -- Original call ---------------------------------------------------
        response = await original_fn(*args, **kwargs)

        # -- Post-call scan --------------------------------------------------
        if should_scan_responses() and response and guard:
            try:
                resp_text = extract_response(response)
                if resp_text:
                    resp_decision = await guard.scan_async(
                        messages=[{"role": "assistant", "content": resp_text}],
                        direction="output",
                        model=model,
                    )
                    if resp_decision.blocked and get_mode() == "enforce":
                        raise PromptGuardBlockedError(resp_decision)
            except (PromptGuardBlockedError, GuardApiError):
                raise
            except Exception:
                logger.debug("Response scanning failed", exc_info=True)

        return response

    return wrapper
