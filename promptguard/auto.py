"""
Auto-instrumentation for PromptGuard.

Call ``promptguard.init()`` once at application startup to automatically
secure LLM calls made through the patched SDK surfaces (see README
"Exact patched call surfaces"), regardless of which framework (LangChain,
CrewAI, AutoGen, LlamaIndex, …) sits on top.

Usage::

    import promptguard
    promptguard.init(api_key="pg_live_xxx")  # or set PROMPTGUARD_API_KEY env var

    # Calls through the patched surfaces are now secured transparently.
    from openai import OpenAI
    client = OpenAI()
    client.chat.completions.create(...)   # ← scanned by PromptGuard

Modes:
    * ``"enforce"``  - block requests that violate policies (default)
    * ``"monitor"``  - log threats but never block (shadow mode)
"""

import logging
import threading
from typing import Any

from promptguard._resolve import resolve_credentials, validate_mode
from promptguard.guard import GuardClient

logger = logging.getLogger("promptguard")

_guard_client: GuardClient | None = None
_mode: str = "enforce"
_fail_open: bool = True
_scan_responses: bool = False
# Guards the module-level globals during init()/shutdown() so concurrent or
# repeat calls don't interleave a half-swapped client.
_state_lock = threading.Lock()


def init(
    api_key: str | None = None,
    base_url: str | None = None,
    mode: str = "enforce",
    fail_open: bool = True,
    scan_responses: bool = False,
    timeout: float = 10.0,
) -> None:
    """Initialise PromptGuard auto-instrumentation.

    Prefer calling this **once at application startup**, before spawning worker
    threads or issuing LLM calls. Re-initialising at runtime is supported and
    thread-safe (the new client is swapped in before the old one is closed), but
    an in-flight request already holding the previous client may see its
    connection closed underneath it.

    Parameters
    ----------
    api_key:
        PromptGuard API key.  Falls back to ``PROMPTGUARD_API_KEY`` env var.
    base_url:
        PromptGuard API base URL.  Falls back to ``PROMPTGUARD_BASE_URL``
        env var, then ``https://api.promptguard.co/api/v1``.
    mode:
        ``"enforce"`` to block policy violations, ``"monitor"`` to log only.
    fail_open:
        If ``True`` (default), allow LLM calls when the Guard API is
        unreachable.  Set to ``False`` to fail-closed.
    scan_responses:
        If ``True``, also scan LLM responses through the Guard API with
        ``direction="output"``.  Defaults to ``False`` here (the zero-config
        drop-in) to avoid doubling Guard API round-trips per LLM call; the
        framework callback handlers default this to ``True`` because they
        already receive response events for free.
    timeout:
        HTTP timeout (seconds) for Guard API calls.  Defaults to ``10.0``: the
        Guard scan is a fast, standalone call in the request path.  (The proxy
        client's default is ``30.0`` because it fronts the full upstream LLM
        call, which is inherently slower.)
    """
    global _guard_client, _mode, _fail_open, _scan_responses

    resolved_key, resolved_url = resolve_credentials(api_key, base_url)
    validate_mode(mode)

    new_client = GuardClient(
        api_key=resolved_key,
        base_url=resolved_url,
        timeout=timeout,
    )

    # Swap the new client (and flags) into place atomically, then close the old
    # one outside the lock. New callers see the new client before we close the
    # old, so we never tear down a client we've just published.
    with _state_lock:
        old_client = _guard_client
        _guard_client = new_client
        _mode = mode
        _fail_open = fail_open
        _scan_responses = scan_responses

    if old_client is not None:
        old_client.close()

    _apply_patches()

    patched = patched_sdks()
    logger.info(
        "PromptGuard auto-instrumentation initialised (mode=%s, fail_open=%s); patched SDKs: %s",
        mode,
        fail_open,
        ", ".join(patched) if patched else "none detected",
    )


def shutdown() -> None:
    """Remove all patches and close the guard client."""
    global _guard_client

    _remove_patches()

    with _state_lock:
        old_client = _guard_client
        _guard_client = None

    if old_client:
        old_client.close()

    logger.info("PromptGuard auto-instrumentation shut down")


# -- Accessor helpers used by individual patch modules -----------------------


def get_guard_client() -> GuardClient | None:
    return _guard_client


def get_mode() -> str:
    return _mode


def is_fail_open() -> bool:
    return _fail_open


def should_scan_responses() -> bool:
    return _scan_responses


# -- Introspection helpers (public API) --------------------------------------


def patched_sdks() -> list[str]:
    """Return the names of the SDKs actually patched by the last ``init()``.

    Useful in tests/health checks to assert instrumentation is live, e.g.::

        promptguard.init(api_key=...)
        assert "openai" in promptguard.patched_sdks()

    Returns an empty list before ``init()`` or after ``shutdown()``, or when no
    supported SDK is importable in the current environment.
    """
    return [patch_module.NAME for patch_module in _applied_patches]


def is_active() -> bool:
    """Return ``True`` when auto-instrumentation is initialised (a guard client
    is installed). Note this is independent of whether any SDK was patchable —
    use :func:`patched_sdks` to confirm specific SDKs are instrumented."""
    return _guard_client is not None


# -- Patch orchestration -----------------------------------------------------

_applied_patches: list = []


def _try_apply_patch(patch_module: Any) -> None:
    try:
        if patch_module.apply():
            _applied_patches.append(patch_module)
            logger.debug("Patched %s", patch_module.NAME)
    except Exception:
        logger.debug(
            "Skipping %s (not installed or incompatible)",
            patch_module.NAME,
            exc_info=True,
        )


def _try_revert_patch(patch_module: Any) -> None:
    try:
        patch_module.revert()
        logger.debug("Reverted %s", patch_module.NAME)
    except Exception:
        logger.warning("Failed to revert %s", patch_module.NAME, exc_info=True)


def _apply_patches() -> None:
    """Patch every supported SDK.  Missing packages are silently skipped."""
    from promptguard.patches import (
        anthropic_patch,
        bedrock_patch,
        cohere_patch,
        google_patch,
        openai_patch,
    )

    for patch_module in [openai_patch, anthropic_patch, google_patch, cohere_patch, bedrock_patch]:
        _try_apply_patch(patch_module)


def _remove_patches() -> None:
    """Revert all applied patches."""
    for patch_module in _applied_patches:
        _try_revert_patch(patch_module)
    _applied_patches.clear()
