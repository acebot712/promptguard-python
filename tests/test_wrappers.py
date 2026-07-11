"""
Direct tests for the shared patch wrappers (``wrap_sync`` / ``wrap_async``).

These exercise the block/redact/allow decision flow, monitor vs enforce,
fail_open / fail_closed, and response scanning with a stubbed guard client —
the security-critical logic that previously had no direct coverage.
"""

import pytest

import promptguard.auto as auto
from promptguard.guard import GuardApiError, GuardDecision, PromptGuardBlockedError
from promptguard.patches._base import wrap_async, wrap_sync


class StubGuard:
    """Minimal guard client stub with programmable scan results."""

    def __init__(
        self,
        input_decision=None,
        output_decision=None,
        input_error=None,
        output_error=None,
    ):
        self.input_decision = input_decision
        self.output_decision = output_decision
        self.input_error = input_error
        self.output_error = output_error
        self.directions: list[str] = []

    def scan(self, messages, direction="input", model=None, context=None):
        self.directions.append(direction)
        if direction == "input":
            if self.input_error is not None:
                raise self.input_error
            return self.input_decision
        if self.output_error is not None:
            raise self.output_error
        return self.output_decision

    async def scan_async(self, messages, direction="input", model=None, context=None):
        return self.scan(messages, direction=direction, model=model, context=context)


def _decision(decision, **extra):
    data = {"decision": decision, "event_id": "evt", "confidence": 0.9}
    data.update(extra)
    return GuardDecision(data)


def _extract_messages(args, kwargs):
    return kwargs.get("messages", []), kwargs.get("model"), {"framework": "test"}


def _extract_response(response):
    return response.get("text") if isinstance(response, dict) else None


def _apply_redaction(args, kwargs, redacted):
    new_kwargs = dict(kwargs)
    new_kwargs["messages"] = redacted
    return new_kwargs


@pytest.fixture
def guard_env(monkeypatch):
    """Install a stub guard client and controllable mode/flags on promptguard.auto."""

    def _install(guard, *, mode="enforce", fail_open=True, scan_responses=False):
        monkeypatch.setattr(auto, "_guard_client", guard)
        monkeypatch.setattr(auto, "_mode", mode)
        monkeypatch.setattr(auto, "_fail_open", fail_open)
        monkeypatch.setattr(auto, "_scan_responses", scan_responses)
        return guard

    return _install


# ── Sync wrapper ────────────────────────────────────────────────────────


class TestWrapSync:
    def test_allow_calls_original(self, guard_env):
        guard_env(StubGuard(input_decision=_decision("allow")))
        seen = {}

        def original(**kwargs):
            seen.update(kwargs)
            return {"ok": True}

        wrapped = wrap_sync(original, _extract_messages, _extract_response)
        result = wrapped(messages=[{"role": "user", "content": "hi"}], model="gpt")
        assert result == {"ok": True}
        assert seen["messages"] == [{"role": "user", "content": "hi"}]

    def test_no_guard_client_passes_through(self, guard_env):
        guard_env(None)

        wrapped = wrap_sync(lambda **k: "raw", _extract_messages, _extract_response)
        assert wrapped(messages=[{"role": "user", "content": "hi"}]) == "raw"

    def test_block_in_enforce_raises(self, guard_env):
        guard_env(StubGuard(input_decision=_decision("block", threat_type="prompt_injection")))
        called = False

        def original(**kwargs):
            nonlocal called
            called = True
            return "should not run"

        wrapped = wrap_sync(original, _extract_messages, _extract_response)
        with pytest.raises(PromptGuardBlockedError):
            wrapped(messages=[{"role": "user", "content": "attack"}])
        assert called is False

    def test_block_in_monitor_passes_through(self, guard_env):
        guard_env(
            StubGuard(input_decision=_decision("block", threat_type="prompt_injection")),
            mode="monitor",
        )
        wrapped = wrap_sync(lambda **k: "ran", _extract_messages, _extract_response)
        assert wrapped(messages=[{"role": "user", "content": "attack"}]) == "ran"

    def test_redact_applied_in_enforce(self, guard_env):
        redacted = [{"role": "user", "content": "My SSN is [REDACTED]"}]
        guard_env(
            StubGuard(input_decision=_decision("redact", redacted_messages=redacted)),
        )
        seen = {}

        def original(**kwargs):
            seen.update(kwargs)
            return "ok"

        wrapped = wrap_sync(original, _extract_messages, _extract_response, _apply_redaction)
        wrapped(messages=[{"role": "user", "content": "My SSN is 123-45-6789"}])
        assert seen["messages"] == redacted

    def test_redact_without_handler_blocks_in_enforce(self, guard_env):
        """Fail safe: redact + no apply_redaction must block, not forward PII."""
        redacted = [{"role": "user", "content": "[REDACTED]"}]
        guard_env(StubGuard(input_decision=_decision("redact", redacted_messages=redacted)))
        called = False

        def original(**kwargs):
            nonlocal called
            called = True
            return "leaked"

        wrapped = wrap_sync(original, _extract_messages, _extract_response)
        with pytest.raises(PromptGuardBlockedError):
            wrapped(messages=[{"role": "user", "content": "My SSN is 123-45-6789"}])
        assert called is False

    def test_redact_missing_payload_blocks_in_enforce(self, guard_env):
        """Redact decision with NO redacted_messages cannot be honored → block."""
        guard_env(StubGuard(input_decision=_decision("redact", threat_type="pii")))
        called = False

        def original(**kwargs):
            nonlocal called
            called = True
            return "leaked"

        wrapped = wrap_sync(original, _extract_messages, _extract_response, _apply_redaction)
        with pytest.raises(PromptGuardBlockedError):
            wrapped(messages=[{"role": "user", "content": "My SSN is 123-45-6789"}])
        assert called is False

    def test_redact_empty_payload_blocks_in_enforce(self, guard_env):
        guard_env(StubGuard(input_decision=_decision("redact", redacted_messages=[])))
        wrapped = wrap_sync(
            lambda **k: "leaked", _extract_messages, _extract_response, _apply_redaction
        )
        with pytest.raises(PromptGuardBlockedError):
            wrapped(messages=[{"role": "user", "content": "My SSN is 123-45-6789"}])

    def test_redact_partial_payload_blocks_in_enforce(self, guard_env):
        """Fewer redacted messages than scanned ones would leave the tail
        unredacted → escalate to block."""
        guard_env(
            StubGuard(
                input_decision=_decision(
                    "redact",
                    redacted_messages=[{"role": "user", "content": "[REDACTED]"}],
                )
            )
        )
        called = False

        def original(**kwargs):
            nonlocal called
            called = True
            return "leaked"

        wrapped = wrap_sync(original, _extract_messages, _extract_response, _apply_redaction)
        with pytest.raises(PromptGuardBlockedError):
            wrapped(
                messages=[
                    {"role": "user", "content": "My SSN is 123-45-6789"},
                    {"role": "user", "content": "My card is 4111-1111-1111-1111"},
                ]
            )
        assert called is False

    def test_redact_partial_payload_monitor_passes_through(self, guard_env):
        guard_env(
            StubGuard(
                input_decision=_decision(
                    "redact",
                    redacted_messages=[{"role": "user", "content": "[REDACTED]"}],
                )
            ),
            mode="monitor",
        )
        seen = {}
        wrapped = wrap_sync(
            lambda **k: seen.update(k), _extract_messages, _extract_response, _apply_redaction
        )
        original_messages = [
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ]
        wrapped(messages=original_messages)
        assert seen["messages"] == original_messages

    def test_redact_applier_returning_none_blocks_in_enforce(self, guard_env):
        """An applier that cannot rewrite the call shape signals None → block."""
        redacted = [{"role": "user", "content": "[REDACTED]"}]
        guard_env(StubGuard(input_decision=_decision("redact", redacted_messages=redacted)))
        called = False

        def original(**kwargs):
            nonlocal called
            called = True
            return "leaked"

        wrapped = wrap_sync(original, _extract_messages, _extract_response, lambda a, k, r: None)
        with pytest.raises(PromptGuardBlockedError):
            wrapped(messages=[{"role": "user", "content": "x"}])
        assert called is False

    def test_redact_extra_payload_applies_in_enforce(self, guard_env):
        """More redacted messages than scanned is fine (extras ignored)."""
        redacted = [
            {"role": "user", "content": "[REDACTED]"},
            {"role": "user", "content": "extra"},
        ]
        guard_env(StubGuard(input_decision=_decision("redact", redacted_messages=redacted)))
        seen = {}

        def original(**kwargs):
            seen.update(kwargs)
            return "ok"

        wrapped = wrap_sync(original, _extract_messages, _extract_response, _apply_redaction)
        wrapped(messages=[{"role": "user", "content": "orig"}])
        assert seen["messages"] == redacted

    def test_redact_without_handler_monitor_passes_through(self, guard_env):
        redacted = [{"role": "user", "content": "[REDACTED]"}]
        guard_env(
            StubGuard(input_decision=_decision("redact", redacted_messages=redacted)),
            mode="monitor",
        )
        seen = {}
        wrapped = wrap_sync(lambda **k: seen.update(k), _extract_messages, _extract_response)
        wrapped(messages=[{"role": "user", "content": "orig"}])
        # Original (unmodified) messages forwarded in monitor mode.
        assert seen["messages"] == [{"role": "user", "content": "orig"}]

    def test_fail_open_on_api_error(self, guard_env):
        guard_env(StubGuard(input_error=GuardApiError("down", 503)), fail_open=True)
        wrapped = wrap_sync(lambda **k: "allowed", _extract_messages, _extract_response)
        assert wrapped(messages=[{"role": "user", "content": "hi"}]) == "allowed"

    def test_fail_closed_on_api_error(self, guard_env):
        guard_env(StubGuard(input_error=GuardApiError("down", 503)), fail_open=False)
        wrapped = wrap_sync(lambda **k: "allowed", _extract_messages, _extract_response)
        with pytest.raises(GuardApiError):
            wrapped(messages=[{"role": "user", "content": "hi"}])

    def test_response_scan_blocks_in_enforce(self, guard_env):
        guard_env(
            StubGuard(
                input_decision=_decision("allow"),
                output_decision=_decision("block", threat_type="pii_leak"),
            ),
            scan_responses=True,
        )
        wrapped = wrap_sync(lambda **k: {"text": "leak"}, _extract_messages, _extract_response)
        with pytest.raises(PromptGuardBlockedError):
            wrapped(messages=[{"role": "user", "content": "hi"}])

    def test_response_scan_allows_when_clean(self, guard_env):
        guard = guard_env(
            StubGuard(input_decision=_decision("allow"), output_decision=_decision("allow")),
            scan_responses=True,
        )
        wrapped = wrap_sync(lambda **k: {"text": "fine"}, _extract_messages, _extract_response)
        assert wrapped(messages=[{"role": "user", "content": "hi"}]) == {"text": "fine"}
        assert "output" in guard.directions

    def test_response_scan_api_error_fail_open_allows(self, guard_env):
        """A guard outage during response scanning honors fail_open=True."""
        guard_env(
            StubGuard(
                input_decision=_decision("allow"),
                output_error=GuardApiError("down", 503),
            ),
            scan_responses=True,
            fail_open=True,
        )
        wrapped = wrap_sync(lambda **k: {"text": "resp"}, _extract_messages, _extract_response)
        assert wrapped(messages=[{"role": "user", "content": "hi"}]) == {"text": "resp"}

    def test_response_scan_api_error_fail_closed_raises(self, guard_env):
        guard_env(
            StubGuard(
                input_decision=_decision("allow"),
                output_error=GuardApiError("down", 503),
            ),
            scan_responses=True,
            fail_open=False,
        )
        wrapped = wrap_sync(lambda **k: {"text": "resp"}, _extract_messages, _extract_response)
        with pytest.raises(GuardApiError):
            wrapped(messages=[{"role": "user", "content": "hi"}])

    def test_should_intercept_false_skips_scan(self, guard_env):
        guard = guard_env(StubGuard(input_decision=_decision("block")))
        wrapped = wrap_sync(
            lambda **k: "raw",
            _extract_messages,
            _extract_response,
            should_intercept=lambda a, k: False,
        )
        assert wrapped(messages=[{"role": "user", "content": "x"}]) == "raw"
        assert guard.directions == []


# ── Async wrapper ───────────────────────────────────────────────────────


class TestWrapAsync:
    @pytest.mark.asyncio
    async def test_allow_calls_original(self, guard_env):
        guard_env(StubGuard(input_decision=_decision("allow")))

        async def original(**kwargs):
            return {"ok": True}

        wrapped = wrap_async(original, _extract_messages, _extract_response)
        assert await wrapped(messages=[{"role": "user", "content": "hi"}]) == {"ok": True}

    @pytest.mark.asyncio
    async def test_block_in_enforce_raises(self, guard_env):
        guard_env(StubGuard(input_decision=_decision("block")))
        called = False

        async def original(**kwargs):
            nonlocal called
            called = True

        wrapped = wrap_async(original, _extract_messages, _extract_response)
        with pytest.raises(PromptGuardBlockedError):
            await wrapped(messages=[{"role": "user", "content": "x"}])
        assert called is False

    @pytest.mark.asyncio
    async def test_redact_applied_in_enforce(self, guard_env):
        redacted = [{"role": "user", "content": "[REDACTED]"}]
        guard_env(StubGuard(input_decision=_decision("redact", redacted_messages=redacted)))
        seen = {}

        async def original(**kwargs):
            seen.update(kwargs)

        wrapped = wrap_async(original, _extract_messages, _extract_response, _apply_redaction)
        await wrapped(messages=[{"role": "user", "content": "orig"}])
        assert seen["messages"] == redacted

    @pytest.mark.asyncio
    async def test_redact_missing_payload_blocks_in_enforce(self, guard_env):
        guard_env(StubGuard(input_decision=_decision("redact", threat_type="pii")))
        called = False

        async def original(**kwargs):
            nonlocal called
            called = True
            return "leaked"

        wrapped = wrap_async(original, _extract_messages, _extract_response, _apply_redaction)
        with pytest.raises(PromptGuardBlockedError):
            await wrapped(messages=[{"role": "user", "content": "My SSN is 123-45-6789"}])
        assert called is False

    @pytest.mark.asyncio
    async def test_redact_partial_payload_blocks_in_enforce(self, guard_env):
        guard_env(
            StubGuard(
                input_decision=_decision(
                    "redact",
                    redacted_messages=[{"role": "user", "content": "[REDACTED]"}],
                )
            )
        )

        async def original(**kwargs):
            return "leaked"

        wrapped = wrap_async(original, _extract_messages, _extract_response, _apply_redaction)
        with pytest.raises(PromptGuardBlockedError):
            await wrapped(
                messages=[
                    {"role": "user", "content": "one"},
                    {"role": "user", "content": "two"},
                ]
            )

    @pytest.mark.asyncio
    async def test_redact_without_handler_blocks(self, guard_env):
        redacted = [{"role": "user", "content": "[REDACTED]"}]
        guard_env(StubGuard(input_decision=_decision("redact", redacted_messages=redacted)))

        async def original(**kwargs):
            return "leaked"

        wrapped = wrap_async(original, _extract_messages, _extract_response)
        with pytest.raises(PromptGuardBlockedError):
            await wrapped(messages=[{"role": "user", "content": "x"}])

    @pytest.mark.asyncio
    async def test_fail_open_on_api_error(self, guard_env):
        guard_env(StubGuard(input_error=GuardApiError("down", 503)), fail_open=True)

        async def original(**kwargs):
            return "allowed"

        wrapped = wrap_async(original, _extract_messages, _extract_response)
        assert await wrapped(messages=[{"role": "user", "content": "hi"}]) == "allowed"

    @pytest.mark.asyncio
    async def test_fail_closed_on_api_error(self, guard_env):
        guard_env(StubGuard(input_error=GuardApiError("down", 503)), fail_open=False)

        async def original(**kwargs):
            return "allowed"

        wrapped = wrap_async(original, _extract_messages, _extract_response)
        with pytest.raises(GuardApiError):
            await wrapped(messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_response_scan_api_error_fail_open_allows(self, guard_env):
        guard_env(
            StubGuard(
                input_decision=_decision("allow"),
                output_error=GuardApiError("down", 503),
            ),
            scan_responses=True,
            fail_open=True,
        )

        async def original(**kwargs):
            return {"text": "resp"}

        wrapped = wrap_async(original, _extract_messages, _extract_response)
        assert await wrapped(messages=[{"role": "user", "content": "hi"}]) == {"text": "resp"}

    @pytest.mark.asyncio
    async def test_response_scan_api_error_fail_closed_raises(self, guard_env):
        guard_env(
            StubGuard(
                input_decision=_decision("allow"),
                output_error=GuardApiError("down", 503),
            ),
            scan_responses=True,
            fail_open=False,
        )

        async def original(**kwargs):
            return {"text": "resp"}

        wrapped = wrap_async(original, _extract_messages, _extract_response)
        with pytest.raises(GuardApiError):
            await wrapped(messages=[{"role": "user", "content": "hi"}])
