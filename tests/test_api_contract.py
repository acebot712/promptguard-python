"""
Contract tests that validate the Python SDK against api-contract.json.

These tests ensure that error parsing, request field names, and response
field expectations stay in sync with the backend API.  If a test here
fails, it means the SDK and the backend have drifted apart.
"""

import json
from pathlib import Path

import httpx
import pytest

from promptguard._version import __version__
from promptguard.client import _parse_error, _sdk_headers

CONTRACT = json.loads((Path(__file__).parent / "api-contract.json").read_text())


# ── Error parsing ──────────────────────────────────────────────────────────


class TestErrorContract:
    """_parse_error must extract all fields the contract specifies."""

    @pytest.fixture(params=CONTRACT["error_responses"]["cases"], ids=lambda c: c["name"])
    def error_case(self, request):
        return request.param

    def test_error_parsing(self, error_case):
        if "expect_has_detail" in error_case:
            pytest.skip("validation errors use a different envelope")

        body = error_case["body"]
        status = error_case["status_code"]
        expect = error_case["expect"]

        response = httpx.Response(
            status_code=status,
            json=body,
            request=httpx.Request("POST", "http://test"),
        )
        err = _parse_error(response)

        assert err.status_code == status
        assert err.message == expect["message"]
        assert err.code == expect["code"]

        if "type" in expect:
            assert err.error_type == expect["type"]
        if "upgrade_url" in expect:
            assert err.upgrade_url == expect["upgrade_url"]
        if "current_plan" in expect:
            assert err.current_plan == expect["current_plan"]
        if "requests_used" in expect:
            assert err.requests_used == expect["requests_used"]
        if "requests_limit" in expect:
            assert err.requests_limit == expect["requests_limit"]


# ── SDK headers ────────────────────────────────────────────────────────────


class TestHeaderContract:
    """SDK must send the headers the contract requires."""

    def test_required_headers_present(self):
        headers = _sdk_headers("pg_live_test_key")
        for name in CONTRACT["sdk_headers"]["required_headers"]:
            assert name in headers, f"Missing required header: {name}"

    def test_api_key_header(self):
        key = "pg_live_test_key"
        headers = _sdk_headers(key)
        assert headers["X-API-Key"] == key

    def test_recommended_headers_present(self):
        headers = _sdk_headers("pg_live_test_key")
        for name in CONTRACT["sdk_headers"]["recommended_headers"]:
            assert name in headers, f"Missing recommended header: {name}"

    def test_sdk_version_matches_package(self):
        headers = _sdk_headers("pg_live_test_key")
        assert headers["X-PromptGuard-Version"] == __version__


# ── Scan request/response field names ──────────────────────────────────────


class TestScanContract:
    """Scan request/response field names must match the contract."""

    def test_request_uses_content_not_text(self):
        scan = CONTRACT["security_scan"]
        assert "content" in scan["request_fields"]["required"]
        assert "text" not in scan["request_fields"]["required"]

    def test_response_has_required_fields(self):
        scan = CONTRACT["security_scan"]
        required = set(scan["response_fields"]["required"])
        for case in scan["cases"]:
            response_keys = set(case["response"].keys())
            missing = required - response_keys
            assert not missing, f"Case '{case['name']}' missing: {missing}"


# ── Redact request/response field names ────────────────────────────────────


class TestRedactContract:
    """Redact endpoint uses 'original', 'redacted', 'piiFound'."""

    def test_request_uses_content_not_text(self):
        redact = CONTRACT["security_redact"]
        assert "content" in redact["request_fields"]["required"]
        assert "text" not in redact["request_fields"]["required"]

    def test_response_uses_correct_field_names(self):
        redact = CONTRACT["security_redact"]
        required = set(redact["response_fields"]["required"])
        assert required == {"original", "redacted", "piiFound"}
