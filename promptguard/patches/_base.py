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

import copy as _copy
import functools
import logging
from collections.abc import Callable
from typing import Any

from promptguard.guard import GuardApiError, PromptGuardBlockedError

logger = logging.getLogger("promptguard")


# -- Shared decision helpers (used by both sync and async wrappers) ----------


def rewrite_message_object(msg: Any, attr: str, value: str) -> Any | None:
    """Return a copy of the message object ``msg`` with ``attr`` replaced.

    Used by redaction handlers to rewrite attribute-based message objects
    (e.g. Cohere ``ChatMessage`` pydantic models) instead of silently
    forwarding their original (unredacted) content.  Tries, in order:

    1. pydantic v2 ``model_copy(update=...)``
    2. pydantic v1 ``copy(update=...)``
    3. ``copy.copy`` + ``setattr``

    Returns ``None`` when the object cannot be rewritten; callers must treat
    that as "redaction cannot be applied" (block in enforce mode).
    """
    model_copy = getattr(msg, "model_copy", None)
    if callable(model_copy):
        try:
            return model_copy(update={attr: value})
        except Exception:
            logger.debug("model_copy() rewrite failed", exc_info=True)
    copy_method = getattr(msg, "copy", None)
    if callable(copy_method):
        try:
            return copy_method(update={attr: value})
        except Exception:
            logger.debug("copy(update=...) rewrite failed", exc_info=True)
    try:
        clone = _copy.copy(msg)
        setattr(clone, attr, value)
    except Exception:
        logger.debug("copy.copy() + setattr rewrite failed", exc_info=True)
        return None
    return clone


def _handle_pre_scan_decision(
    decision: Any,
    guard_messages: list[dict[str, str]],
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

    if decision.redacted:
        redacted_messages = decision.redacted_messages or []
        # A redacted_messages list SHORTER than the scanned guard messages
        # would leave the unmatched trailing messages unredacted — the
        # per-SDK redaction handlers keep original content for guard indices
        # past the end of the list.  Treat a partial list like a missing one.
        is_partial = len(redacted_messages) < len(guard_messages)
        if get_mode() == "enforce":
            new_kwargs: dict | None = None
            if redacted_messages and not is_partial and apply_redaction is not None:
                # Handlers return ``None`` when the call shape cannot be
                # rewritten safely; that escalates to a block below.
                new_kwargs = apply_redaction(args, kwargs, redacted_messages)
            if new_kwargs is None:
                # Fail safe: the redact decision cannot be honored — the
                # Guard API returned no/empty/partial redacted messages, this
                # SDK has no redaction handler, or the handler could not
                # rewrite the message shape.  Forwarding the ORIGINAL
                # (unredacted) content would leak the exact content the guard
                # flagged, so we block instead.
                logger.error(
                    "PromptGuard: redact decision could not be applied "
                    "(%d redacted messages for %d scanned); blocking the "
                    "request to avoid forwarding unredacted content "
                    "(threat=%s, event=%s)",
                    len(redacted_messages),
                    len(guard_messages),
                    decision.threat_type,
                    decision.event_id,
                )
                raise PromptGuardBlockedError(decision)
            return new_kwargs
        logger.warning(
            "[monitor] PromptGuard would redact%s: %s (event=%s)",
            (
                f" (partial: {len(redacted_messages)} redacted messages "
                f"for {len(guard_messages)} scanned)"
                if is_partial
                else ""
            ),
            decision.threat_type,
            decision.event_id,
        )

    return kwargs


def _handle_response_block(resp_decision: Any, get_mode: Callable[[], str]) -> None:
    """Raise if a response scan decision blocks in enforce mode.

    In monitor mode we log a symmetric warning (mirroring the input path) so a
    blocked *output* is still visible in shadow mode rather than silently
    passing through.
    """
    if not resp_decision.blocked:
        return
    if get_mode() == "enforce":
        raise PromptGuardBlockedError(resp_decision)
    logger.warning(
        "[monitor] PromptGuard would block response: %s (event=%s)",
        resp_decision.threat_type,
        resp_decision.event_id,
    )


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
        Optional ``(args, kwargs, redacted_messages) -> kwargs | None``.
        Return ``None`` when the call shape cannot be rewritten safely; in
        enforce mode that escalates the redact decision to a block.
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

            kwargs = _handle_pre_scan_decision(
                decision, guard_messages, get_mode, apply_redaction, args, kwargs
            )

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
            except PromptGuardBlockedError:
                raise
            except GuardApiError:
                # Mirror the input path: a guard outage during response scanning
                # honors fail_open instead of unconditionally re-raising.
                if not is_fail_open():
                    raise
                logger.warning(
                    "Guard API unavailable during response scan, allowing (fail_open=True)"
                )
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

            kwargs = _handle_pre_scan_decision(
                decision, guard_messages, get_mode, apply_redaction, args, kwargs
            )

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
            except PromptGuardBlockedError:
                raise
            except GuardApiError:
                # Mirror the input path: a guard outage during response scanning
                # honors fail_open instead of unconditionally re-raising.
                if not is_fail_open():
                    raise
                logger.warning(
                    "Guard API unavailable during response scan, allowing (fail_open=True)"
                )
            except Exception:
                logger.debug("Response scanning failed", exc_info=True)

        return response

    return wrapper
