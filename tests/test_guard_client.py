"""
Tests for the GuardClient and GuardDecision classes.
"""

from unittest.mock import MagicMock, patch

import pytest

from promptguard.guard import (
    GuardApiError,
    GuardClient,
    GuardDecision,
    PromptGuardBlockedError,
)


class TestGuardDecision:
    """Test GuardDecision data class."""

    def test_allow_decision(self):
        d = GuardDecision({"decision": "allow", "confidence": 1.0, "event_id": "abc"})
        assert d.allowed
        assert not d.blocked
        assert not d.redacted

    def test_block_decision(self):
        d = GuardDecision(
            {
                "decision": "block",
                "confidence": 0.95,
                "event_id": "xyz",
                "threat_type": "prompt_injection",
            }
        )
        assert d.blocked
        assert not d.allowed
        assert not d.redacted
        assert d.threat_type == "prompt_injection"

    def test_redact_decision(self):
        d = GuardDecision(
            {
                "decision": "redact",
                "confidence": 0.9,
                "event_id": "def",
                "threat_type": "pii_leak",
                "redacted_messages": [{"role": "user", "content": "[REDACTED]"}],
            }
        )
        assert d.redacted
        assert not d.blocked
        assert d.redacted_messages is not None
        assert len(d.redacted_messages) == 1

    def test_defaults_for_missing_optional_fields(self):
        d = GuardDecision({"decision": "allow"})
        assert d.decision == "allow"
        assert d.confidence == 0.0
        assert d.event_id == ""
        assert d.threat_type is None
        assert d.threats == []

    def test_missing_decision_is_rejected(self):
        # Contract v1.4.0: malformed/empty bodies must not default to allow.
        with pytest.raises(GuardApiError):
            GuardDecision({})

    def test_unknown_decision_is_rejected(self):
        with pytest.raises(GuardApiError):
            GuardDecision({"decision": "maybe"})


class TestPromptGuardBlockedError:
    """Test the blocked error exception."""

    def test_error_message(self):
        d = GuardDecision(
            {
                "decision": "block",
                "confidence": 0.95,
                "event_id": "test-123",
                "threat_type": "prompt_injection",
            }
        )
        error = PromptGuardBlockedError(d)
        assert "prompt_injection" in str(error)
        assert "test-123" in str(error)
        assert error.decision is d

    def test_error_without_threat_type(self):
        d = GuardDecision({"decision": "block", "event_id": "test"})
        error = PromptGuardBlockedError(d)
        assert "policy_violation" in str(error)


class TestGuardClient:
    """Test the GuardClient HTTP wrapper."""

    def test_init_defaults(self, monkeypatch):
        monkeypatch.delenv("PROMPTGUARD_BASE_URL", raising=False)
        client = GuardClient(api_key="pg_test_key")
        assert client._api_key == "pg_test_key"
        assert client._guard_url == "https://api.promptguard.co/api/v1/guard"

    def test_init_missing_key_raises_actionable_error(self, monkeypatch):
        # No positional key and no env var → the shared actionable ValueError,
        # not a bare TypeError about a missing positional argument.
        monkeypatch.delenv("PROMPTGUARD_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            GuardClient()

    def test_init_falls_back_to_env_key(self, monkeypatch):
        monkeypatch.setenv("PROMPTGUARD_API_KEY", "pg_env_key")
        monkeypatch.delenv("PROMPTGUARD_BASE_URL", raising=False)
        client = GuardClient()
        assert client._api_key == "pg_env_key"

    def test_init_custom_url(self):
        client = GuardClient(
            api_key="pg_test",
            base_url="http://localhost:8080/api/v1",
        )
        assert client._guard_url == "http://localhost:8080/api/v1/guard"

    def test_init_strips_trailing_slash(self):
        client = GuardClient(
            api_key="pg_test",
            base_url="http://localhost:8080/api/v1/",
        )
        assert client._guard_url == "http://localhost:8080/api/v1/guard"

    def test_build_payload_minimal(self):
        payload = GuardClient._build_payload(
            messages=[{"role": "user", "content": "Hello"}],
            direction="input",
            model=None,
            context=None,
        )
        assert payload == {
            "messages": [{"role": "user", "content": "Hello"}],
            "direction": "input",
        }

    def test_build_payload_full(self):
        payload = GuardClient._build_payload(
            messages=[{"role": "user", "content": "Hello"}],
            direction="output",
            model="gpt-5-nano",
            context={"framework": "langchain"},
        )
        assert payload["model"] == "gpt-5-nano"
        assert payload["context"]["framework"] == "langchain"
        assert payload["direction"] == "output"

    def test_headers(self):
        client = GuardClient(api_key="pg_test_key")
        headers = client._get_headers()
        assert headers["X-API-Key"] == "pg_test_key"
        assert headers["Content-Type"] == "application/json"
        assert "X-PromptGuard-SDK" in headers

    @patch("promptguard.guard.httpx.Client")
    def test_scan_success(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "decision": "allow",
            "event_id": "evt-1",
            "confidence": 1.0,
            "latency_ms": 5.0,
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = GuardClient(api_key="pg_test")
        result = client.scan(
            messages=[{"role": "user", "content": "Hello"}],
            direction="input",
        )

        assert result.allowed
        assert result.event_id == "evt-1"

    @patch("promptguard.guard.httpx.Client")
    def test_scan_api_error_raises_guard_api_error(self, mock_client_cls):
        """Guard API errors should raise GuardApiError so callers decide."""
        from promptguard.guard import GuardApiError

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = GuardClient(api_key="pg_test")
        with pytest.raises(GuardApiError) as exc_info:
            client.scan(
                messages=[{"role": "user", "content": "Hello"}],
                direction="input",
            )
        assert exc_info.value.status_code == 500

    @patch("promptguard.guard.httpx.Client")
    def test_scan_network_error_raises_guard_api_error(self, mock_client_cls):
        """Network errors should raise GuardApiError so callers decide."""
        from promptguard.guard import GuardApiError

        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("Connection refused")
        mock_client_cls.return_value = mock_client

        client = GuardClient(api_key="pg_test")
        with pytest.raises(GuardApiError):
            client.scan(
                messages=[{"role": "user", "content": "Hello"}],
                direction="input",
            )


class TestAsyncClientPerLoop:
    """Async clients are kept per event loop: a second live loop must not
    displace (and thereby kill in-flight scans of) the first loop's client."""

    def test_two_live_loops_get_separate_stable_clients(self):
        """Threads each running asyncio.run get their own stable client."""
        import asyncio
        import threading

        guard = GuardClient(api_key="pg_test")
        barrier = threading.Barrier(2)
        results: dict[str, tuple] = {}
        errors: list[BaseException] = []

        def run(name: str) -> None:
            async def main() -> None:
                first = guard._ensure_async_client()
                # Both loops hold a client at the same time before re-asking.
                barrier.wait(timeout=10)
                second = guard._ensure_async_client()
                barrier.wait(timeout=10)
                third = guard._ensure_async_client()
                results[name] = (first, second, third)
                await first.aclose()

            try:
                asyncio.run(main())
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(n,)) for n in ("a", "b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors
        # Stable within each loop: no rebuild/displacement across the other
        # loop's concurrent use, so in-flight scans are never killed.
        for name in ("a", "b"):
            assert results[name][0] is results[name][1]
            assert results[name][1] is results[name][2]
        # Distinct across loops.
        assert results["a"][0] is not results["b"][0]

    def test_ensure_async_client_outside_loop_raises(self):
        guard = GuardClient(api_key="pg_test")
        with pytest.raises(GuardApiError):
            guard._ensure_async_client()

    def test_sync_context_manager_closes_client(self):
        """`with GuardClient(...)` closes the sync client on exit."""
        with GuardClient(api_key="pg_test") as guard:
            assert guard is not None
            client = guard._ensure_sync_client()
            assert not client.is_closed
        assert client.is_closed

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_client(self):
        """`async with GuardClient(...)` closes the current loop's client."""
        async with GuardClient(api_key="pg_test") as guard:
            client = guard._ensure_async_client()
            assert not client.is_closed
        assert client.is_closed

    def test_close_closes_every_cached_client(self):
        """close() drains the per-loop map and closes each client."""
        import asyncio

        guard = GuardClient(api_key="pg_test")

        async def build():
            return guard._ensure_async_client()

        # Keep the loops referenced so GC-eviction doesn't close the clients
        # before close() does.
        loop_a = asyncio.new_event_loop()
        loop_b = asyncio.new_event_loop()
        try:
            client_a = loop_a.run_until_complete(build())
            client_b = loop_b.run_until_complete(build())
            assert client_a is not client_b
            assert not client_a.is_closed
            assert not client_b.is_closed

            guard.close()

            assert client_a.is_closed
            assert client_b.is_closed
            assert len(guard._async_clients) == 0
        finally:
            loop_a.close()
            loop_b.close()

    # The explicit gc.collect() below also finalizes garbage left behind by
    # unrelated earlier tests; don't let their ResourceWarnings fail this one.
    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    def test_loop_gc_closes_evicted_client(self):
        """A garbage-collected loop's client is best-effort closed on eviction."""
        import asyncio
        import gc

        guard = GuardClient(api_key="pg_test")

        async def build():
            return guard._ensure_async_client()

        client = asyncio.run(build())
        gc.collect()
        assert client.is_closed
        assert len(guard._async_clients) == 0

    @pytest.mark.asyncio
    async def test_aclose_closes_current_loop_client(self):
        guard = GuardClient(api_key="pg_test")
        client = guard._ensure_async_client()
        await guard.aclose()
        assert client.is_closed
        assert len(guard._async_clients) == 0

    def test_schedule_aclose_noop_when_loop_not_running(self):
        loop = MagicMock()
        loop.is_closed.return_value = False
        loop.is_running.return_value = False
        GuardClient._schedule_aclose(MagicMock(), loop)
        loop.call_soon_threadsafe.assert_not_called()

    def test_schedule_aclose_noop_when_loop_none(self):
        # No loop → nothing to schedule, no error.
        GuardClient._schedule_aclose(MagicMock(), None)
