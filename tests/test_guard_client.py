"""
Tests for the GuardClient and GuardDecision classes.
"""

from unittest.mock import MagicMock, patch

import pytest

from promptguard.guard import GuardClient, GuardDecision, PromptGuardBlockedError


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

    def test_defaults_for_missing_fields(self):
        d = GuardDecision({})
        assert d.decision == "allow"
        assert d.confidence == 0.0
        assert d.event_id == ""
        assert d.threat_type is None
        assert d.threats == []


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
            model="gpt-4o",
            context={"framework": "langchain"},
        )
        assert payload["model"] == "gpt-4o"
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
