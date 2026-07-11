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

    def test_init_defaults(self):
        client = GuardClient(api_key="pg_test_key")
        assert client._api_key == "pg_test_key"
        assert client._guard_url == "https://api.promptguard.co/api/v1/guard"

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


class TestAsyncClientLoopRebuild:
    """The async client is loop-bound; a loop change must not leak the old one."""

    def test_loop_change_schedules_close_of_displaced_client(self):
        client = GuardClient(api_key="pg_test")

        old_client = MagicMock()
        old_loop = MagicMock()
        old_loop.is_closed.return_value = False
        old_loop.is_running.return_value = True
        client._async_client = old_client
        client._async_loop = old_loop

        # Simulate a *different* running loop without spinning up real sockets.
        new_loop = MagicMock()
        with (
            patch("promptguard.guard.asyncio.get_running_loop", return_value=new_loop),
            patch("promptguard.guard.httpx.AsyncClient", return_value=MagicMock()) as mock_ac,
        ):
            rebuilt = client._ensure_async_client()

        assert rebuilt is not old_client
        assert client._async_loop is new_loop
        mock_ac.assert_called_once()
        # The displaced client's close was scheduled on its original loop.
        old_loop.call_soon_threadsafe.assert_called_once()

    def test_schedule_aclose_noop_when_loop_not_running(self):
        loop = MagicMock()
        loop.is_closed.return_value = False
        loop.is_running.return_value = False
        GuardClient._schedule_aclose(MagicMock(), loop)
        loop.call_soon_threadsafe.assert_not_called()

    def test_schedule_aclose_noop_when_loop_none(self):
        # No loop → nothing to schedule, no error.
        GuardClient._schedule_aclose(MagicMock(), None)
