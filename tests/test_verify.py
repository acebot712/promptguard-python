"""`verify()` has to report the failures that otherwise look like success.

The SDK fails open, so the states worth testing are the quiet ones: a key that
is rejected, a Guard API that cannot be reached, and a policy that lets a plain
injection through. In every one of those an instrumented application keeps
serving traffic and blocks nothing, which is exactly why a positive check has to
exist and has to be honest about what it found.

The distinction these tests pin down is fail vs warn. A request that never
completed is a **fail** — the integration is broken. A request that completed
and came back permissive is a **warn** — the integration works and the policy is
the thing to look at. Collapsing the two in either direction makes the result
useless: all-fail cries wolf on a monitor-mode project, all-warn hides a dead
API key.
"""

from __future__ import annotations

import httpx
import pytest

from promptguard import _verify as verify_module
from promptguard._verify import verify
from promptguard.client import PromptGuardError


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("PROMPTGUARD_API_KEY", "pg_live_test")
    monkeypatch.setenv("PROMPTGUARD_BASE_URL", "https://api.example.test/api/v1")


@pytest.fixture(autouse=True)
def _quiet_instrumentation(monkeypatch):
    """Keep the instrumentation check out of the way unless a test wants it.

    Without this the result depends on which provider SDKs happen to be
    installed in the environment running the suite.
    """
    monkeypatch.setattr(
        verify_module,
        "instrumentation_report",
        lambda: {"patched": ["openai"], "detected_unpatched": [], "advice_url": "https://x.test"},
    )


class _FakeSecurity:
    def __init__(self, scan_result=None, scan_error=None, redact_result=None, redact_error=None):
        self._scan_result = scan_result if scan_result is not None else {"blocked": True}
        self._scan_error = scan_error
        self._redact_result = (
            redact_result if redact_result is not None else {"piiFound": ["email", "ssn"]}
        )
        self._redact_error = redact_error

    def scan(self, content, content_type="prompt"):
        if self._scan_error:
            raise self._scan_error
        return self._scan_result

    def redact(self, content, pii_types=None):
        if self._redact_error:
            raise self._redact_error
        return self._redact_result


class _FakeConfig:
    """The real client rewrites base_url to add the `/proxy` suffix."""

    base_url = "https://api.example.test/api/v1/proxy"


class _FakeClient:
    def __init__(self, security):
        self.security = security
        self.config = _FakeConfig()
        self.closed = False

    def close(self):
        self.closed = True


def _patch_client(monkeypatch, security) -> _FakeClient:
    client = _FakeClient(security)
    monkeypatch.setattr(verify_module, "PromptGuard", lambda **kwargs: client)
    return client


def _status(result, name):
    return next(c["status"] for c in result["checks"] if c["name"] == name)


class TestHealthyIntegration:
    def test_everything_working_reports_ok(self, monkeypatch):
        _patch_client(monkeypatch, _FakeSecurity())

        result = verify()

        assert result["ok"] is True
        assert result["checks_failed"] == 0
        assert _status(result, "connectivity") == "pass"
        assert _status(result, "authentication") == "pass"
        assert _status(result, "threat_detection") == "pass"
        assert _status(result, "pii_redaction") == "pass"

    def test_the_documented_one_line_assertion_holds(self, monkeypatch):
        """The README tells people to write `assert promptguard.verify()["ok"]`."""
        _patch_client(monkeypatch, _FakeSecurity())
        assert verify()["ok"]

    def test_the_client_is_closed_even_though_checks_never_raise(self, monkeypatch):
        client = _patch_client(monkeypatch, _FakeSecurity(scan_error=httpx.ConnectError("boom")))
        verify()
        assert client.closed is True


class TestBrokenIntegration:
    """These are the states that otherwise look exactly like a working app."""

    def test_a_rejected_api_key_fails_rather_than_warns(self, monkeypatch):
        _patch_client(
            monkeypatch,
            _FakeSecurity(
                scan_error=PromptGuardError(
                    message="Invalid API key", code="UNAUTHORIZED", status_code=401
                )
            ),
        )

        result = verify()

        assert result["ok"] is False
        assert _status(result, "authentication") == "fail"
        # The host answered — it answered 401 — so connectivity is fine and
        # saying otherwise would send someone debugging their network.
        assert _status(result, "connectivity") == "pass"

    def test_an_unreachable_guard_api_fails_every_dependent_check(self, monkeypatch):
        _patch_client(monkeypatch, _FakeSecurity(scan_error=httpx.ConnectError("no route to host")))

        result = verify()

        assert result["ok"] is False
        assert _status(result, "connectivity") == "fail"
        assert _status(result, "authentication") == "fail"
        assert _status(result, "threat_detection") == "fail"
        # Not attempted, and reported as not attempted rather than as a pass.
        assert _status(result, "pii_redaction") == "fail"

    def test_a_failing_redact_call_does_not_take_down_the_scan_result(self, monkeypatch):
        _patch_client(
            monkeypatch,
            _FakeSecurity(redact_error=PromptGuardError("boom", "SERVER_ERROR", 500)),
        )

        result = verify()

        assert result["ok"] is False
        assert _status(result, "pii_redaction") == "fail"
        assert _status(result, "threat_detection") == "pass"


class TestPermissivePolicy:
    """The request worked; the answer was not what a protected setup gives."""

    def test_an_unblocked_injection_warns_without_clearing_ok(self, monkeypatch):
        _patch_client(monkeypatch, _FakeSecurity(scan_result={"blocked": False}))

        result = verify()

        assert _status(result, "threat_detection") == "warn"
        assert result["checks_warned"] >= 1
        # Monitor-mode projects legitimately do not block. Failing here would
        # make verify() unusable for them, and a check people must ignore is a
        # check they stop reading.
        assert result["ok"] is True

    def test_undetected_pii_warns(self, monkeypatch):
        _patch_client(monkeypatch, _FakeSecurity(redact_result={"piiFound": []}))

        result = verify()

        assert _status(result, "pii_redaction") == "warn"
        assert result["ok"] is True

    def test_a_missing_piifound_key_is_not_read_as_success(self, monkeypatch):
        """Guards the field name. `piiFound` is camelCase in the API response;
        spelling it `pii_found` here would silently warn on every healthy
        integration, which is the same silent-wrongness this module exists to
        prevent."""
        _patch_client(monkeypatch, _FakeSecurity(redact_result={"redacted": "..."}))

        assert _status(verify(), "pii_redaction") == "warn"


class TestInstrumentationCheck:
    def test_an_installed_but_unhooked_provider_warns_by_name(self, monkeypatch):
        monkeypatch.setattr(
            verify_module,
            "instrumentation_report",
            lambda: {
                "patched": ["openai"],
                "detected_unpatched": ["google-genai"],
                "advice_url": "https://x.test",
            },
        )
        _patch_client(monkeypatch, _FakeSecurity())

        result = verify()

        check = next(c for c in result["checks"] if c["name"] == "instrumentation")
        assert check["status"] == "warn"
        assert "google-genai" in check["detail"]

    def test_no_patches_warns_but_does_not_fail(self, monkeypatch):
        """Proxy-mode users are fully protected with zero patches applied."""
        monkeypatch.setattr(
            verify_module,
            "instrumentation_report",
            lambda: {"patched": [], "detected_unpatched": [], "advice_url": "https://x.test"},
        )
        _patch_client(monkeypatch, _FakeSecurity())

        result = verify()

        assert _status(result, "instrumentation") == "warn"
        assert result["ok"] is True


class TestCallerErrors:
    def test_a_missing_api_key_raises_rather_than_reporting_a_failed_check(self, monkeypatch):
        """A caller who never supplied a key has a bug, not a finding."""
        monkeypatch.delenv("PROMPTGUARD_API_KEY", raising=False)

        with pytest.raises(ValueError, match="API key required"):
            verify()


def test_the_report_names_the_url_that_was_actually_called(monkeypatch):
    """A result pasted into an issue has to say what it checked.

    The client rewrites the base URL to append `/proxy`, so reporting the value
    that was passed in would name a URL no request ever went to.
    """
    _patch_client(monkeypatch, _FakeSecurity())

    result = verify()

    assert result["base_url"] == "https://api.example.test/api/v1/proxy"
    assert result["sdk_version"]
    assert result["instrumentation"]["patched"] == ["openai"]
