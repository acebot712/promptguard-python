"""
Guard client - Calls the PromptGuard Guard API for content scanning.

This is used internally by auto-instrumentation patches and framework
integrations. It sends messages to POST /api/v1/guard and returns the
decision (allow / block / redact).
"""

import asyncio
import logging
import threading
from typing import Any

import httpx

from promptguard._version import __version__

logger = logging.getLogger("promptguard")


class GuardDecision:
    """Result of a guard API call."""

    __slots__ = (
        "confidence",
        "decision",
        "event_id",
        "latency_ms",
        "redacted_messages",
        "threat_type",
        "threats",
    )

    def __init__(self, data: dict[str, Any]):
        self.decision: str = data.get("decision", "allow")
        self.event_id: str = data.get("event_id", "")
        self.confidence: float = data.get("confidence", 0.0)
        self.threat_type: str | None = data.get("threat_type")
        self.redacted_messages: list[dict[str, str]] | None = data.get("redacted_messages")
        self.threats: list[dict[str, Any]] = data.get("threats", [])
        self.latency_ms: float = data.get("latency_ms", 0.0)

    @property
    def blocked(self) -> bool:
        return self.decision == "block"

    @property
    def redacted(self) -> bool:
        return self.decision == "redact"

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"


class PromptGuardBlockedError(Exception):
    """Raised when PromptGuard blocks a request in enforce mode."""

    def __init__(self, decision: GuardDecision):
        self.decision = decision
        threat = decision.threat_type or "policy_violation"
        super().__init__(
            f"PromptGuard blocked this request: {threat} "
            f"(confidence={decision.confidence:.2f}, event_id={decision.event_id})"
        )


class GuardApiError(Exception):
    """Raised when the Guard API is unreachable or returns an error.

    Only surfaced when ``fail_open=False``.  When ``fail_open=True``
    (the default), callers catch this and return an allow decision.
    """

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class GuardClient:
    """HTTP client for the PromptGuard Guard API.

    Provides both sync and async methods.  The client itself never
    swallows errors; callers (patches, integrations) decide whether
    to fail open or closed based on configuration.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.promptguard.co/api/v1",
        timeout: float = 10.0,
    ):
        self._api_key = api_key
        self._guard_url = f"{base_url.rstrip('/')}/guard"
        self._timeout = timeout
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None
        # The async client is bound to the event loop it was created on; track
        # it so we can detect a loop change and rebuild.
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._client_lock = threading.Lock()

    def _get_headers(self) -> dict[str, str]:
        return {
            "X-API-Key": self._api_key,
            "Content-Type": "application/json",
            "X-PromptGuard-SDK": "python-auto",
            "X-PromptGuard-Version": __version__,
        }

    def _ensure_sync_client(self) -> httpx.Client:
        # Double-checked locking: avoid two threads each building a client and
        # leaking one.
        if self._sync_client is None:
            with self._client_lock:
                if self._sync_client is None:
                    self._sync_client = httpx.Client(timeout=self._timeout)
        return self._sync_client

    def _ensure_async_client(self) -> httpx.AsyncClient:
        try:
            current_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        with self._client_lock:
            # A client created on a now-defunct loop can't be reused; drop it
            # (best effort — we can't await aclose() here) and rebuild.
            if (
                self._async_client is not None
                and current_loop is not None
                and self._async_loop is not current_loop
            ):
                logger.debug("Event loop changed; rebuilding async Guard client")
                self._async_client = None
            if self._async_client is None:
                self._async_client = httpx.AsyncClient(timeout=self._timeout)
                self._async_loop = current_loop
        return self._async_client

    def _check_response(self, resp: httpx.Response) -> GuardDecision:
        """Validate response status and parse into a GuardDecision.

        A malformed 200 body (non-JSON, or JSON that isn't an object, or an
        unknown ``decision``) is converted to a ``GuardApiError`` so callers'
        fail-open / fail-closed handling applies instead of an unhandled
        ``JSONDecodeError`` / ``AttributeError`` escaping the wrapper.
        """
        if resp.status_code >= 400:
            raise GuardApiError(
                f"Guard API returned {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code,
            )
        try:
            data = resp.json()
        except (ValueError, TypeError) as exc:
            raise GuardApiError(
                f"Guard API returned a non-JSON body: {exc}",
                status_code=resp.status_code,
            ) from exc
        if not isinstance(data, dict):
            raise GuardApiError(
                f"Guard API returned an unexpected body type: {type(data).__name__}",
                status_code=resp.status_code,
            )
        decision = GuardDecision(data)
        if decision.decision not in ("allow", "block", "redact"):
            raise GuardApiError(
                f"Guard API returned an unknown decision: {decision.decision!r}",
                status_code=resp.status_code,
            )
        return decision

    def scan(
        self,
        messages: list[dict[str, str]],
        direction: str = "input",
        model: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> GuardDecision:
        """Synchronous scan via the guard API."""
        payload = self._build_payload(messages, direction, model, context)
        client = self._ensure_sync_client()
        try:
            resp = client.post(self._guard_url, json=payload, headers=self._get_headers())
        except Exception as exc:
            raise GuardApiError(f"Guard API call failed: {exc}") from exc
        return self._check_response(resp)

    async def scan_async(
        self,
        messages: list[dict[str, str]],
        direction: str = "input",
        model: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> GuardDecision:
        """Asynchronous scan via the guard API."""
        payload = self._build_payload(messages, direction, model, context)
        client = self._ensure_async_client()
        try:
            resp = await client.post(self._guard_url, json=payload, headers=self._get_headers())
        except Exception as exc:
            raise GuardApiError(f"Guard API call failed: {exc}") from exc
        return self._check_response(resp)

    @staticmethod
    def _build_payload(
        messages: list[dict[str, str]],
        direction: str,
        model: str | None,
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": messages,
            "direction": direction,
        }
        if model:
            payload["model"] = model
        if context:
            payload["context"] = context
        return payload

    def close(self):
        """Close the sync client and best-effort close the async client.

        Safe to call from sync code (e.g. ``init()`` re-init and
        ``shutdown()``): the async client is closed on its own loop if one is
        running, otherwise via a throwaway loop, and any failure is swallowed.
        """
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
        self._best_effort_close_async()

    def _best_effort_close_async(self) -> None:
        if self._async_client is None:
            return
        client = self._async_client
        self._async_client = None
        self._async_loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            loop.create_task(client.aclose())
            return
        try:
            asyncio.run(client.aclose())
        except Exception:
            logger.debug("Best-effort async Guard client close failed", exc_info=True)

    async def aclose(self):
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
            self._async_loop = None
