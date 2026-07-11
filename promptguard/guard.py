"""
Guard client - Calls the PromptGuard Guard API for content scanning.

This is used internally by auto-instrumentation patches and framework
integrations. It sends messages to POST /api/v1/guard and returns the
decision (allow / block / redact).
"""

import asyncio
import logging
import threading
import weakref
from typing import Any

import httpx

from promptguard._resolve import resolve_credentials
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
        decision = data.get("decision")
        if decision not in ("allow", "block", "redact"):
            # Contract v1.4.0: a malformed/empty body must NOT silently
            # default to allow. Raise the API-error type so the caller's
            # explicit fail-open / fail-closed policy governs instead.
            raise GuardApiError(f"Guard API returned an unknown decision: {decision!r}")
        self.decision: str = decision
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


async def _silent_aclose(client: httpx.AsyncClient) -> None:
    """Await ``client.aclose()``, swallowing any failure (best-effort)."""
    try:
        await client.aclose()
    except Exception:
        logger.debug("Async Guard client close failed", exc_info=True)


def _best_effort_close_orphan(client: httpx.AsyncClient) -> None:
    """Best-effort close of a client whose event loop is gone.

    Called from ``close()`` and from the ``weakref.finalize`` hook when a
    loop is garbage-collected.  Runs the aclose on the current loop if one
    is running here, otherwise on a throwaway loop; failures are swallowed.
    """
    if client.is_closed:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    try:
        if loop is not None and loop.is_running():
            loop.create_task(_silent_aclose(client))
        else:
            asyncio.run(client.aclose())
    except Exception:
        logger.debug("Best-effort async Guard client close failed", exc_info=True)


class GuardClient:
    """HTTP client for the PromptGuard Guard API.

    Provides both sync and async methods.  The client itself never
    swallows errors; callers (patches, integrations) decide whether
    to fail open or closed based on configuration.

    ``timeout`` defaults to ``10.0`` seconds — the Guard scan is a fast,
    standalone call.  (The proxy client / ``Config`` default is ``30.0`` because
    it fronts the full upstream LLM call.)  ``api_key`` / ``base_url`` fall back
    to ``PROMPTGUARD_API_KEY`` / ``PROMPTGUARD_BASE_URL`` when omitted.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 10.0,
    ):
        # Route through the shared resolver so a missing key falls back to
        # PROMPTGUARD_API_KEY and, if still absent, raises the same actionable
        # ValueError as every other entry point (rather than a bare TypeError).
        resolved_key, resolved_url = resolve_credentials(api_key, base_url)
        self._api_key = resolved_key
        self._guard_url = f"{resolved_url.rstrip('/')}/guard"
        self._timeout = timeout
        self._sync_client: httpx.Client | None = None
        # An httpx.AsyncClient is bound to the event loop it was created on,
        # so keep one client per live loop.  Two loops scanning concurrently
        # (e.g. threads each running asyncio.run) each get their own client
        # instead of displacing — and killing in-flight scans of — the other.
        # Weak keys let a garbage-collected loop drop its entry; the paired
        # ``weakref.finalize`` best-effort closes the evicted client.
        self._async_clients: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, httpx.AsyncClient
        ] = weakref.WeakKeyDictionary()
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
        """Return the async client bound to the *current* running loop.

        Each live event loop gets its own client (per-loop map), so a scan on
        loop B never closes a client with in-flight scans on loop A.
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            # scan_async() is always awaited inside a loop; anything else is
            # a caller bug that must not silently build an unbound client.
            raise GuardApiError("GuardClient async API used outside a running event loop") from exc
        with self._client_lock:
            client = self._async_clients.get(current_loop)
            if client is None or client.is_closed:
                client = httpx.AsyncClient(timeout=self._timeout)
                try:
                    self._async_clients[current_loop] = client
                    # Best-effort close when the loop itself is evicted
                    # (garbage-collected) so its connection pool isn't leaked.
                    weakref.finalize(current_loop, _best_effort_close_orphan, client)
                except TypeError:
                    # Exotic non-weakrefable loop: fall back to an uncached
                    # client (closed by the httpx finalizer / GC).
                    logger.debug("Event loop is not weak-referenceable; client not cached")
        return client

    @staticmethod
    def _schedule_aclose(client: httpx.AsyncClient, loop: asyncio.AbstractEventLoop | None) -> None:
        """Best-effort close of ``client`` on the ``loop`` it was bound to.

        Only attempts anything when that loop is still alive and running; the
        aclose() is scheduled thread-safely so it runs inside that loop.
        """
        if loop is None or loop.is_closed() or not loop.is_running():
            return
        try:
            loop.call_soon_threadsafe(lambda: loop.create_task(_silent_aclose(client)))
        except RuntimeError:
            # Loop stopped/closed between the check and the schedule.
            logger.debug("Could not schedule aclose on client's loop", exc_info=True)

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
        # GuardDecision itself rejects unknown decisions (contract v1.4.0);
        # attach the HTTP status for callers that inspect it.
        try:
            return GuardDecision(data)
        except GuardApiError as exc:
            raise GuardApiError(str(exc), status_code=resp.status_code) from exc

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

    def __enter__(self) -> "GuardClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> "GuardClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def close(self):
        """Close the sync client and best-effort close all async clients.

        Safe to call from sync code (e.g. ``init()`` re-init and
        ``shutdown()``): each async client is closed on the loop it belongs
        to if that loop is still running, otherwise via a throwaway loop, and
        any failure is swallowed.
        """
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
        for loop, client in self._drain_async_clients():
            self._best_effort_close_async(client, loop)

    def _drain_async_clients(
        self,
    ) -> list[tuple[asyncio.AbstractEventLoop, httpx.AsyncClient]]:
        """Atomically take ownership of every cached (loop, client) pair."""
        with self._client_lock:
            items = list(self._async_clients.items())
            self._async_clients.clear()
        return items

    def _best_effort_close_async(
        self,
        client: httpx.AsyncClient,
        loop: asyncio.AbstractEventLoop | None,
    ) -> None:
        """Close ``client`` on its own ``loop`` when possible, else orphan-close."""
        if loop is not None and not loop.is_closed() and loop.is_running():
            self._schedule_aclose(client, loop)
            return
        _best_effort_close_orphan(client)

    async def aclose(self):
        """Async close: awaits the current loop's client, best-effort for others."""
        try:
            current_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        for loop, client in self._drain_async_clients():
            if loop is current_loop:
                await _silent_aclose(client)
            else:
                self._best_effort_close_async(client, loop)
