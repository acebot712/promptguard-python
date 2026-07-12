"""
Shared contract tests - validates the Python SDK against guard-contract.json.

If this test fails, the Python SDK has drifted from the cross-SDK
contract. Fix the SDK, not the contract (unless both SDKs agree on
the change).
"""

import json
from pathlib import Path

import pytest

CONTRACT_PATH = Path(__file__).resolve().parent / "guard-contract.json"


@pytest.fixture(scope="module")
def contract():
    assert CONTRACT_PATH.exists(), f"Contract file not found: {CONTRACT_PATH}"
    with CONTRACT_PATH.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# GuardDecision
# ---------------------------------------------------------------------------


class TestGuardDecisionContract:
    def test_all_cases(self, contract):
        from promptguard.guard import GuardApiError, GuardDecision

        error_types = {"GuardApiError": GuardApiError}

        for case in contract["guard_decision"]["cases"]:
            if "expect_error" in case:
                with pytest.raises(error_types[case["expect_error"]]):
                    GuardDecision(case["input"])
                continue

            decision = GuardDecision(case["input"])
            expect = case["expect"]

            assert decision.allowed == expect["allowed"], f"{case['name']}: allowed"
            assert decision.blocked == expect["blocked"], f"{case['name']}: blocked"
            assert decision.redacted == expect["redacted"], f"{case['name']}: redacted"
            assert decision.event_id == expect["event_id"], f"{case['name']}: event_id"
            assert decision.confidence == expect["confidence"], f"{case['name']}: confidence"
            assert decision.threat_type == expect.get("threat_type"), f"{case['name']}: threat_type"

            if "redacted_messages_count" in expect:
                assert len(decision.redacted_messages) == expect["redacted_messages_count"]
                assert decision.redacted_messages[0]["content"] == expect["redacted_first_content"]


# ---------------------------------------------------------------------------
# Redaction enforcement (v1.5.0)
# ---------------------------------------------------------------------------


class TestRedactionEnforcementContract:
    """Drives the central decision handling (``wrap_sync``) with each contract
    case.  Input direction: a redact decision with missing/empty/partial
    redacted_messages must escalate to block in enforce mode.  Output
    direction (``direction: "output"``, v1.5.1): a redact decision on the LLM
    response can never be applied and must block in enforce mode."""

    def test_all_cases(self, contract, monkeypatch):
        import promptguard.auto as auto
        from promptguard.guard import GuardDecision, PromptGuardBlockedError
        from promptguard.patches._base import wrap_sync

        for case in contract["redaction_enforcement"]["cases"]:
            decision = GuardDecision(case["decision"])
            scanned = case["scanned_messages"]
            direction = case.get("direction", "input")

            class StubGuard:
                def __init__(self, d):
                    self._d = d

                def scan(self, messages, direction="input", model=None, context=None):
                    return self._d

            monkeypatch.setattr(auto, "_guard_client", StubGuard(decision))
            monkeypatch.setattr(auto, "_mode", case["mode"])
            monkeypatch.setattr(auto, "_fail_open", True)
            monkeypatch.setattr(auto, "_scan_responses", direction == "output")

            forwarded: dict = {}

            applier = None
            if case["has_redaction_applier"]:

                def applier(args, kwargs, redacted):
                    new_kwargs = dict(kwargs)
                    new_kwargs["messages"] = redacted
                    return new_kwargs

            if direction == "output":
                # Drive the response-scan path: no input messages, the
                # original returns the scanned assistant content as response.
                response_text = scanned[0]["content"]

                def original(*, forwarded=forwarded, response_text=response_text, **kwargs):
                    forwarded.update(kwargs)
                    return {"text": response_text}

                def extract(args, kwargs):
                    return [], None, {"framework": "contract"}

                def extract_response(response):
                    return response.get("text")

                wrapped = wrap_sync(original, extract, extract_response, applier)

                if case["expect"] == "block":
                    with pytest.raises(PromptGuardBlockedError):
                        wrapped(messages=scanned)
                elif case["expect"] == "passthrough":
                    result = wrapped(messages=scanned)
                    assert result == {"text": response_text}, (
                        f"{case['name']}: original response not returned"
                    )
                else:  # pragma: no cover - contract drift guard
                    raise AssertionError(
                        f"{case['name']}: unknown output expect {case['expect']!r}"
                    )
                continue

            def original(*, forwarded=forwarded, **kwargs):
                forwarded.update(kwargs)
                return "ok"

            def extract(args, kwargs, scanned=scanned):
                return scanned, None, {"framework": "contract"}

            wrapped = wrap_sync(original, extract, lambda response: None, applier)

            if case["expect"] == "block":
                with pytest.raises(PromptGuardBlockedError):
                    wrapped(messages=scanned)
                assert forwarded == {}, f"{case['name']}: original ran despite block"
            elif case["expect"] == "apply":
                wrapped(messages=scanned)
                assert forwarded["messages"] == case["decision"]["redacted_messages"], (
                    f"{case['name']}: redacted content not forwarded"
                )
            elif case["expect"] == "passthrough":
                wrapped(messages=scanned)
                assert forwarded["messages"] == scanned, (
                    f"{case['name']}: original content not passed through"
                )
            else:  # pragma: no cover - contract drift guard
                raise AssertionError(f"{case['name']}: unknown expect {case['expect']!r}")


# ---------------------------------------------------------------------------
# OpenAI message conversion
# ---------------------------------------------------------------------------


class TestOpenAIMessageContract:
    def test_all_cases(self, contract):
        from promptguard.patches.openai_patch import _messages_to_guard_format

        for case in contract["message_conversion"]["cases"]:
            result = _messages_to_guard_format(case["input"])
            assert result == case["expect"], f"{case['name']}: mismatch"


# ---------------------------------------------------------------------------
# Anthropic message conversion
# ---------------------------------------------------------------------------


class TestAnthropicMessageContract:
    def test_all_cases(self, contract):
        from promptguard.patches.anthropic_patch import _messages_to_guard_format

        for case in contract["anthropic_message_conversion"]["cases"]:
            result = _messages_to_guard_format(
                case["input_messages"],
                system=case["input_system"],
            )
            assert result == case["expect"], f"{case['name']}: mismatch"


# ---------------------------------------------------------------------------
# Google content conversion
# ---------------------------------------------------------------------------


class TestGoogleContentContract:
    def test_all_cases(self, contract):
        from promptguard.patches.google_patch import _content_to_guard_format

        for case in contract["google_content_conversion"]["cases"]:
            result = _content_to_guard_format(case["input"])
            assert result == case["expect"], f"{case['name']}: mismatch"


# ---------------------------------------------------------------------------
# PromptGuardBlockedError
# ---------------------------------------------------------------------------


class TestBlockedErrorContract:
    def test_all_cases(self, contract):
        from promptguard.guard import GuardDecision, PromptGuardBlockedError

        for case in contract["blocked_error"]["cases"]:
            decision = GuardDecision(case["decision"])
            error = PromptGuardBlockedError(decision)

            for fragment in case["expect_message_contains"]:
                assert fragment in str(error), (
                    f'{case["name"]}: expected "{fragment}" in error message: {error}'
                )


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


class TestPayloadContract:
    def test_all_cases(self, contract):
        from promptguard.guard import GuardClient

        for case in contract["guard_request_payload"]["cases"]:
            args = case["args"]
            payload = GuardClient._build_payload(
                messages=args["messages"],
                direction=args["direction"],
                model=args.get("model"),
                context=args.get("context"),
            )
            assert payload == case["expect"], f"{case['name']}: mismatch"
