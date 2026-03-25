"""
Auto-instrumentation for PromptGuard.

Call ``promptguard.init()`` once at application startup to automatically
secure *all* LLM calls made through popular SDKs, regardless of which
framework (LangChain, CrewAI, AutoGen, LlamaIndex, …) sits on top.

Usage::

    import promptguard
    promptguard.init(api_key="pg_xxx")  # or set PROMPTGUARD_API_KEY env var

    # Everything below is now secured transparently.
    from openai import OpenAI
    client = OpenAI()
    client.chat.completions.create(...)   # ← scanned by PromptGuard

Modes:
    * ``"enforce"``  - block requests that violate policies (default)
    * ``"monitor"``  - log threats but never block (shadow mode)
"""

import logging
import os
from typing import Any

from promptguard.guard import GuardClient

logger = logging.getLogger("promptguard")

# Global state managed by init() / shutdown().
_guard_client: GuardClient | None = None
_mode: str = "enforce"
_fail_open: bool = True
_scan_responses: bool = False
_initialized: bool = False


def init(
    api_key: str | None = None,
    base_url: str | None = None,
    mode: str = "enforce",
    fail_open: bool = True,
    scan_responses: bool = False,
    timeout: float = 10.0,
) -> None:
    """Initialise PromptGuard auto-instrumentation.

    This monkey-patches the ``create()`` methods on popular LLM SDKs so
    every call is scanned by the PromptGuard Guard API before (and
    optionally after) the LLM is invoked.

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
        ``direction="output"``.
    timeout:
        HTTP timeout (seconds) for Guard API calls.
    """
    global _guard_client, _mode, _fail_open, _scan_responses, _initialized

    resolved_key = api_key or os.environ.get("PROMPTGUARD_API_KEY", "")
    if not resolved_key:
        raise ValueError(
            "PromptGuard API key required. Pass api_key= or set the "
            "PROMPTGUARD_API_KEY environment variable."
        )

    resolved_url = (
        base_url or os.environ.get("PROMPTGUARD_BASE_URL") or "https://api.promptguard.co/api/v1"
    )

    if mode not in ("enforce", "monitor"):
        raise ValueError("mode must be 'enforce' or 'monitor'")

    _guard_client = GuardClient(
        api_key=resolved_key,
        base_url=resolved_url,
        timeout=timeout,
    )
    _mode = mode
    _fail_open = fail_open
    _scan_responses = scan_responses
    _initialized = True

    # Apply patches for every SDK that is already imported (or importable).
    _apply_patches()

    logger.info(
        "PromptGuard auto-instrumentation initialised (mode=%s, fail_open=%s)",
        mode,
        fail_open,
    )


def shutdown() -> None:
    """Remove all patches and close the guard client."""
    global _guard_client, _initialized

    _remove_patches()

    if _guard_client:
        _guard_client.close()
        _guard_client = None

    _initialized = False
    logger.info("PromptGuard auto-instrumentation shut down")


# -- Accessor helpers used by individual patch modules -----------------------


def get_guard_client() -> GuardClient | None:
    """Return the global GuardClient (or None if not initialised)."""
    return _guard_client


def get_mode() -> str:
    return _mode


def is_fail_open() -> bool:
    return _fail_open


def should_scan_responses() -> bool:
    return _scan_responses


def is_initialized() -> bool:
    return _initialized


# -- Patch orchestration -----------------------------------------------------

_applied_patches: list = []


def _try_apply_patch(patch_module: Any) -> None:
    """Attempt to apply a single patch module, logging failures."""
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
    """Attempt to revert a single patch module, logging failures."""
    try:
        patch_module.revert()
        logger.debug("Reverted %s", patch_module.NAME)
    except Exception:
        logger.warning("Failed to revert %s", patch_module.NAME, exc_info=True)


def _apply_patches() -> None:
    """Try to patch every supported SDK.  Missing packages are silently
    skipped; we only patch what is available."""
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
