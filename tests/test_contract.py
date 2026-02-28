"""
Shared contract tests — validates the Python SDK against the shared
fixture file at packages/sdk-shared/guard-contract.json.

If this test fails, the Python SDK has drifted from the cross-SDK
contract. Fix the SDK, not the contract (unless both SDKs agree on
the change).
"""

import json
from pathlib import Path

import pytest

# Locate the shared contract file relative to this test file.
CONTRACT_PATH = Path(__file__).resolve().parent.parent.parent / "sdk-shared" / "guard-contract.json"


@pytest.fixture(scope="module")
def contract():
    assert CONTRACT_PATH.exists(), f"Contract file not found: {CONTRACT_PATH}"
    with open(CONTRACT_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# GuardDecision
# ---------------------------------------------------------------------------


class TestGuardDecisionContract:
    def test_all_cases(self, contract):
        from promptguard.guard import GuardDecision

        for case in contract["guard_decision"]["cases"]:
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
