"""
Tests for the remaining audit fixes: guard response validation, Config
masking/clamping/timeout, config-path base_url normalization, and the new
Bedrock/Cohere redaction handlers.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from promptguard.client import PromptGuard
from promptguard.config import Config
from promptguard.guard import GuardApiError, GuardClient

# ── Guard response validation (malformed 200 bodies) ──────────────────────


def _guard_with_response(mock_client_cls, *, status=200, json_value=None, json_exc=None):
    mock_response = MagicMock()
    mock_response.status_code = status
    if json_exc is not None:
        mock_response.json.side_effect = json_exc
    else:
        mock_response.json.return_value = json_value
    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client_cls.return_value = mock_client
    return GuardClient(api_key="pg_test")


class TestGuardResponseValidation:
    @patch("promptguard.guard.httpx.Client")
    def test_non_json_body_raises_guard_api_error(self, mock_cls):
        client = _guard_with_response(mock_cls, json_exc=ValueError("no json"))
        with pytest.raises(GuardApiError):
            client.scan(messages=[{"role": "user", "content": "hi"}])

    @patch("promptguard.guard.httpx.Client")
    def test_array_body_raises_guard_api_error(self, mock_cls):
        client = _guard_with_response(mock_cls, json_value=["not", "a", "dict"])
        with pytest.raises(GuardApiError):
            client.scan(messages=[{"role": "user", "content": "hi"}])

    @patch("promptguard.guard.httpx.Client")
    def test_unknown_decision_raises_guard_api_error(self, mock_cls):
        client = _guard_with_response(mock_cls, json_value={"decision": "explode"})
        with pytest.raises(GuardApiError):
            client.scan(messages=[{"role": "user", "content": "hi"}])

    @patch("promptguard.guard.httpx.Client")
    def test_valid_decision_passes(self, mock_cls):
        client = _guard_with_response(mock_cls, json_value={"decision": "allow"})
        assert client.scan(messages=[{"role": "user", "content": "hi"}]).allowed


# ── Config ────────────────────────────────────────────────────────────────


class TestConfig:
    def test_repr_masks_api_key(self):
        c = Config(api_key="pg_live_supersecret_value")
        text = repr(c)
        assert "supersecret" not in text
        assert "pg_liv" in text  # a short prefix is fine
        assert "…" in text

    def test_repr_masks_short_key(self):
        assert "***" in repr(Config(api_key="short"))

    def test_repr_masks_medium_key_fully(self):
        # A 12-char key is not long enough to reveal a prefix/suffix.
        text = repr(Config(api_key="pg_live_1234"))  # 12 chars
        assert "***" in text
        assert "pg_liv" not in text

    def test_repr_reveals_only_long_key(self):
        # 13 chars → prefix/suffix shown.
        text = repr(Config(api_key="pg_live_12345"))
        assert "pg_liv" in text
        assert "…" in text

    def test_post_init_clamps_negative(self):
        c = Config(api_key="x", max_retries=-4, retry_delay=-2.0, timeout=-1.0)
        assert c.max_retries == 0
        assert c.retry_delay == 0.0
        assert c.timeout == 0.0

    def test_timeout_field_default(self):
        assert Config(api_key="x").timeout == 30.0


class TestConfigPathNormalization:
    def test_config_base_url_gets_proxy_suffix(self):
        cfg = Config(api_key="pg_test", base_url="https://api.promptguard.co/api/v1")
        pg = PromptGuard(config=cfg)
        assert pg.config.base_url.endswith("/proxy")

    def test_config_api_key_env_fallback(self, monkeypatch):
        monkeypatch.setenv("PROMPTGUARD_API_KEY", "pg_from_env")
        cfg = Config(api_key="")
        pg = PromptGuard(config=cfg)
        assert pg.config.api_key == "pg_from_env"

    def test_config_timeout_honored(self):
        cfg = Config(api_key="pg_test", timeout=7.0)
        pg = PromptGuard(config=cfg)
        assert pg._http.timeout.read == 7.0

    def test_caller_config_not_mutated(self):
        cfg = Config(api_key="pg_test", base_url="https://api.promptguard.co/api/v1")
        original_base = cfg.base_url
        pg = PromptGuard(config=cfg)
        # The caller's Config object is left untouched; the client holds a copy.
        assert cfg.base_url == original_base
        assert pg.config is not cfg
        assert pg.config.base_url.endswith("/proxy")


class TestPathParamEncoding:
    def test_agent_stats_path_is_encoded(self):
        captured = {}

        class _Rec:
            def _request(self, method, path, **kwargs):
                captured["path"] = path
                return {}

        from promptguard.client import Agent

        Agent(_Rec()).stats("weird/../id space")
        assert captured["path"] == "/agent/weird%2F..%2Fid%20space/stats"

    def test_redteam_test_name_is_encoded(self):
        captured = {}

        class _Rec:
            def _request(self, method, path, **kwargs):
                captured["path"] = path
                return {}

        from promptguard.client import RedTeam

        RedTeam(_Rec()).run_test("a/b c")
        assert captured["path"] == "/internal/redteam/test/a%2Fb%20c"


# ── Bedrock redaction ─────────────────────────────────────────────────────


class TestBedrockRedaction:
    def test_invoke_body_bytes_redacted(self):
        from promptguard.patches.bedrock_patch import _apply_redaction

        body = json.dumps(
            {"messages": [{"role": "user", "content": "My SSN is 123-45-6789"}]}
        ).encode()
        api_params = {"modelId": "anthropic.claude", "body": body}
        redacted = [{"role": "user", "content": "My SSN is [REDACTED]"}]

        _apply_redaction((object(), "InvokeModel", api_params), {}, redacted)

        parsed = json.loads(api_params["body"])
        assert parsed["messages"][0]["content"] == "My SSN is [REDACTED]"
        assert isinstance(api_params["body"], bytes)

    def test_invoke_body_with_system_offset(self):
        from promptguard.patches.bedrock_patch import _apply_redaction

        body = json.dumps(
            {"system": "orig sys", "messages": [{"role": "user", "content": "orig user"}]}
        ).encode()
        api_params = {"body": body}
        redacted = [
            {"role": "system", "content": "clean sys"},
            {"role": "user", "content": "clean user"},
        ]
        _apply_redaction((object(), "InvokeModel", api_params), {}, redacted)
        parsed = json.loads(api_params["body"])
        assert parsed["system"] == "clean sys"
        assert parsed["messages"][0]["content"] == "clean user"

    def test_converse_params_redacted(self):
        from promptguard.patches.bedrock_patch import _apply_redaction

        # Real botocore Converse uses lowercase "messages" with block-shaped
        # content ([{"text": ...}]); redaction must write the same shape back.
        api_params = {
            "messages": [{"role": "user", "content": [{"text": "leak"}]}],
        }
        redacted = [{"role": "user", "content": "clean"}]
        _apply_redaction((object(), "Converse", api_params), {}, redacted)
        assert api_params["messages"][0]["content"] == [{"text": "clean"}]

    def test_converse_params_system_block_shaped(self):
        from promptguard.patches.bedrock_patch import _apply_redaction

        api_params = {
            "system": [{"text": "orig sys"}],
            "messages": [{"role": "user", "content": [{"text": "orig user"}]}],
        }
        redacted = [
            {"role": "system", "content": "clean sys"},
            {"role": "user", "content": "clean user"},
        ]
        _apply_redaction((object(), "Converse", api_params), {}, redacted)
        assert api_params["system"] == [{"text": "clean sys"}]
        assert api_params["messages"][0]["content"] == [{"text": "clean user"}]

    def test_converse_params_capitalized_messages_fallback(self):
        from promptguard.patches.bedrock_patch import _apply_redaction

        # Legacy capitalized "Messages" key is still handled (block-shaped).
        api_params = {
            "Messages": [{"role": "user", "content": [{"text": "leak"}]}],
        }
        redacted = [{"role": "user", "content": "clean"}]
        _apply_redaction((object(), "Converse", api_params), {}, redacted)
        assert api_params["Messages"][0]["content"] == [{"text": "clean"}]

    def test_titan_input_text_redacted(self):
        from promptguard.patches.bedrock_patch import _apply_redaction

        api_params = {"body": json.dumps({"inputText": "secret"}).encode()}
        _apply_redaction(
            (object(), "InvokeModel", api_params), {}, [{"role": "user", "content": "clean"}]
        )
        assert json.loads(api_params["body"])["inputText"] == "clean"

    def test_non_dict_entry_does_not_misalign_converse_redaction(self):
        """Extraction skips non-dict entries; redaction must skip the same
        entries instead of consuming a redacted index positionally —
        otherwise the entry AFTER the non-dict one keeps its ORIGINAL
        flagged content."""
        from promptguard.patches.bedrock_patch import (
            _apply_redaction,
            _extract_messages_from_body,
        )

        api_params = {
            "messages": [
                {"role": "user", "content": [{"text": "first leak"}]},
                "not-a-message-dict",
                {"role": "user", "content": [{"text": "second leak"}]},
            ],
        }
        # Extraction emits exactly two guard messages (non-dict skipped).
        guard_messages = _extract_messages_from_body(api_params)
        assert [m["content"] for m in guard_messages] == ["first leak", "second leak"]

        redacted = [
            {"role": "user", "content": "first clean"},
            {"role": "user", "content": "second clean"},
        ]
        out = _apply_redaction((object(), "Converse", api_params), {}, redacted)
        assert out is not None
        assert api_params["messages"][0]["content"] == [{"text": "first clean"}]
        assert api_params["messages"][1] == "not-a-message-dict"
        assert api_params["messages"][2]["content"] == [{"text": "second clean"}]

    def test_non_dict_entry_does_not_misalign_invoke_redaction(self):
        from promptguard.patches.bedrock_patch import (
            _apply_redaction,
            _extract_messages_from_body,
        )

        body = {
            "system": "sys leak",
            "messages": [
                {"role": "user", "content": "user leak"},
                None,
                {"role": "user", "content": "tail leak"},
            ],
        }
        api_params = {"body": json.dumps(body).encode()}
        guard_messages = _extract_messages_from_body(api_params["body"])
        assert len(guard_messages) == 3  # system + two dict messages

        redacted = [
            {"role": "system", "content": "sys clean"},
            {"role": "user", "content": "user clean"},
            {"role": "user", "content": "tail clean"},
        ]
        out = _apply_redaction((object(), "InvokeModel", api_params), {}, redacted)
        assert out is not None
        parsed = json.loads(api_params["body"])
        assert parsed["system"] == "sys clean"
        assert parsed["messages"][0]["content"] == "user clean"
        assert parsed["messages"][1] is None
        assert parsed["messages"][2]["content"] == "tail clean"

    def test_scanned_entry_without_counterpart_returns_none(self):
        """An emitting entry with no redacted counterpart cannot be rewritten;
        the handler must return None so enforce mode escalates to block."""
        from promptguard.patches.bedrock_patch import _apply_redaction

        api_params = {
            "messages": [
                {"role": "user", "content": [{"text": "one"}]},
                {"role": "user", "content": [{"text": "two"}]},
            ],
        }
        out = _apply_redaction(
            (object(), "Converse", api_params), {}, [{"role": "user", "content": "clean"}]
        )
        assert out is None

    def test_undecodable_invoke_body_returns_none(self):
        from promptguard.patches.bedrock_patch import _apply_redaction

        api_params = {"body": b"\x00not-json"}
        out = _apply_redaction(
            (object(), "InvokeModel", api_params), {}, [{"role": "user", "content": "clean"}]
        )
        assert out is None
        assert api_params["body"] == b"\x00not-json"  # left untouched

    def test_unknown_body_shape_returns_none(self):
        from promptguard.patches.bedrock_patch import _apply_redaction

        api_params = {"someOtherField": "x"}
        out = _apply_redaction(
            (object(), "Converse", api_params), {}, [{"role": "user", "content": "clean"}]
        )
        assert out is None

    def test_non_dict_api_params_returns_none(self):
        from promptguard.patches.bedrock_patch import _apply_redaction

        out = _apply_redaction(
            (object(), "Converse", "not-a-dict"), {}, [{"role": "user", "content": "clean"}]
        )
        assert out is None


# ── Bedrock response extraction (scan_responses=True) ─────────────────────


class _ReadOnceStream:
    """Minimal stand-in for a botocore StreamingBody: readable exactly once."""

    def __init__(self, data: bytes):
        self._data = data
        self.reads = 0

    def read(self) -> bytes:
        self.reads += 1
        data, self._data = self._data, b""
        return data


class TestBedrockResponseExtraction:
    def test_converse_output_message_extracted(self):
        from promptguard.patches.bedrock_patch import _extract_response

        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "hello "}, {"text": "world"}],
                }
            },
            "stopReason": "end_turn",
        }
        assert _extract_response(response) == "hello \nworld"

    def test_converse_empty_content_returns_none(self):
        from promptguard.patches.bedrock_patch import _extract_response

        response = {"output": {"message": {"role": "assistant", "content": []}}}
        assert _extract_response(response) is None

    def test_invoke_anthropic_body_bytes_extracted(self):
        from promptguard.patches.bedrock_patch import _extract_response

        body = json.dumps(
            {"content": [{"type": "text", "text": "the answer"}], "stop_reason": "end_turn"}
        ).encode()
        assert _extract_response({"body": body}) == "the answer"

    def test_invoke_anthropic_streaming_body_restored(self):
        from promptguard.patches.bedrock_patch import _extract_response

        raw = json.dumps({"content": [{"type": "text", "text": "streamed"}]}).encode()
        stream = _ReadOnceStream(raw)
        response = {"body": stream, "contentType": "application/json"}

        assert _extract_response(response) == "streamed"
        # The read-once stream was consumed exactly once and the body was put
        # back so the caller can still read the payload.
        assert stream.reads == 1
        restored = response["body"]
        data = restored.read() if hasattr(restored, "read") else restored
        assert json.loads(data)["content"][0]["text"] == "streamed"

    def test_invoke_anthropic_legacy_completion_extracted(self):
        from promptguard.patches.bedrock_patch import _extract_response

        body = json.dumps({"completion": "legacy text"}).encode()
        assert _extract_response({"body": body}) == "legacy text"

    def test_invoke_titan_results_extracted(self):
        from promptguard.patches.bedrock_patch import _extract_response

        body = json.dumps({"results": [{"outputText": "titan says hi"}]}).encode()
        assert _extract_response({"body": body}) == "titan says hi"

    def test_invoke_llama_generation_extracted(self):
        from promptguard.patches.bedrock_patch import _extract_response

        body = json.dumps({"generation": "llama out"}).encode()
        assert _extract_response({"body": body}) == "llama out"

    def test_invoke_mistral_outputs_extracted(self):
        from promptguard.patches.bedrock_patch import _extract_response

        body = json.dumps({"outputs": [{"text": "mistral out"}]}).encode()
        assert _extract_response({"body": body}) == "mistral out"

    def test_unknown_invoke_shape_returns_none(self):
        from promptguard.patches.bedrock_patch import _extract_response

        assert _extract_response({"body": json.dumps({"weird": 1}).encode()}) is None

    def test_non_json_invoke_body_returns_none(self):
        from promptguard.patches.bedrock_patch import _extract_response

        assert _extract_response({"body": b"not json"}) is None

    def test_non_dict_response_returns_none(self):
        from promptguard.patches.bedrock_patch import _extract_response

        assert _extract_response("nope") is None
        assert _extract_response(None) is None

    def test_response_without_output_or_body_returns_none(self):
        from promptguard.patches.bedrock_patch import _extract_response

        assert _extract_response({"stopReason": "end_turn"}) is None


# ── Cohere redaction ──────────────────────────────────────────────────────


class _ChatMessage:
    """Cohere-v1-style ChatMessage stand-in (attribute-based, copyable)."""

    def __init__(self, role, message):
        self.role = role
        self.message = message


class _V2Message:
    """Cohere-v2-style message object stand-in (role/content attributes)."""

    def __init__(self, role, content):
        self.role = role
        self.content = content


class _PydanticV2Message:
    """Message object exposing pydantic-v2-style model_copy(update=...)."""

    def __init__(self, role, content):
        self.role = role
        self.content = content

    def model_copy(self, update=None):
        clone = _PydanticV2Message(self.role, self.content)
        for key, value in (update or {}).items():
            setattr(clone, key, value)
        return clone


class _FrozenMessage:
    """A message object that cannot be rewritten (read-only content)."""

    __slots__ = ("_content", "_role")

    def __init__(self, role, content):
        object.__setattr__(self, "_role", role)
        object.__setattr__(self, "_content", content)

    @property
    def role(self):
        return self._role

    @property
    def content(self):
        return self._content

    def __setattr__(self, name, value):
        raise AttributeError(f"{name} is read-only")


class TestCohereRedaction:
    def test_v2_messages_redacted(self):
        from promptguard.patches.cohere_patch import _apply_redaction

        kwargs = {"messages": [{"role": "user", "content": "leak"}]}
        redacted = [{"role": "user", "content": "clean"}]
        out = _apply_redaction((), kwargs, redacted)
        assert out["messages"][0]["content"] == "clean"

    def test_message_and_history_redacted(self):
        from promptguard.patches.cohere_patch import _apply_redaction

        kwargs = {
            "chat_history": [{"role": "user", "message": "old leak"}],
            "message": "new leak",
        }
        redacted = [
            {"role": "user", "content": "old clean"},
            {"role": "user", "content": "new clean"},
        ]
        out = _apply_redaction((), kwargs, redacted)
        assert out["chat_history"][0]["message"] == "old clean"
        assert out["message"] == "new clean"

    def test_chat_message_objects_rewritten(self):
        """Idiomatic cohere.ChatMessage history objects must be rewritten,
        not silently forwarded with their original content."""
        from promptguard.patches.cohere_patch import _apply_redaction

        history = [_ChatMessage("USER", "old leak"), _ChatMessage("CHATBOT", "reply leak")]
        kwargs = {"chat_history": history, "message": "new leak"}
        redacted = [
            {"role": "USER", "content": "old clean"},
            {"role": "CHATBOT", "content": "reply clean"},
            {"role": "user", "content": "new clean"},
        ]
        out = _apply_redaction((), kwargs, redacted)
        assert out["chat_history"][0].message == "old clean"
        assert out["chat_history"][1].message == "reply clean"
        assert out["message"] == "new clean"
        # Originals are not mutated (copies are forwarded).
        assert history[0].message == "old leak"

    def test_v2_message_objects_rewritten(self):
        from promptguard.patches.cohere_patch import _apply_redaction

        kwargs = {"messages": [_V2Message("user", "leak")]}
        out = _apply_redaction((), kwargs, [{"role": "user", "content": "clean"}])
        assert out["messages"][0].content == "clean"

    def test_pydantic_style_object_rewritten_via_model_copy(self):
        from promptguard.patches.cohere_patch import _apply_redaction

        kwargs = {"messages": [_PydanticV2Message("user", "leak")]}
        out = _apply_redaction((), kwargs, [{"role": "user", "content": "clean"}])
        assert out["messages"][0].content == "clean"

    def test_roleless_history_entry_does_not_consume_index(self):
        """Extraction skips history entries without a role; redaction must
        skip the same entries so indices stay aligned."""
        from promptguard.patches.cohere_patch import _apply_redaction, _to_guard_messages

        roleless = object()  # no role attr → not extracted
        kwargs = {
            "chat_history": [roleless, {"role": "user", "message": "old leak"}],
            "message": "new leak",
        }
        # Extraction sees exactly two guard messages.
        guard_messages = _to_guard_messages(
            message=kwargs["message"], chat_history=kwargs["chat_history"]
        )
        assert len(guard_messages) == 2

        redacted = [
            {"role": "user", "content": "old clean"},
            {"role": "user", "content": "new clean"},
        ]
        out = _apply_redaction((), kwargs, redacted)
        assert out["chat_history"][0] is roleless
        assert out["chat_history"][1]["message"] == "old clean"
        assert out["message"] == "new clean"

    def test_unrewritable_object_returns_none(self):
        """A message with a redacted counterpart that cannot be rewritten
        must fail the whole redaction (None → block in enforce mode)."""
        from promptguard.patches.cohere_patch import _apply_redaction

        kwargs = {"messages": [_FrozenMessage("user", "leak")]}
        assert _apply_redaction((), kwargs, [{"role": "user", "content": "clean"}]) is None

    def test_missing_counterpart_returns_none(self):
        from promptguard.patches.cohere_patch import _apply_redaction

        kwargs = {
            "messages": [
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
            ]
        }
        assert _apply_redaction((), kwargs, [{"role": "user", "content": "clean"}]) is None

    def test_no_redactable_shape_returns_none(self):
        from promptguard.patches.cohere_patch import _apply_redaction

        assert _apply_redaction((), {}, [{"role": "user", "content": "clean"}]) is None

    def test_v1_preamble_consumes_first_redacted_index(self):
        """The scanned preamble is guard index 0; history and message shift
        by one — redaction must mirror that exactly."""
        from promptguard.patches.cohere_patch import _apply_redaction, _to_guard_messages

        kwargs = {
            "preamble": "sys leak",
            "chat_history": [{"role": "user", "message": "old leak"}],
            "message": "new leak",
        }
        guard_messages = _to_guard_messages(
            message=kwargs["message"],
            chat_history=kwargs["chat_history"],
            preamble=kwargs["preamble"],
        )
        assert len(guard_messages) == 3

        redacted = [
            {"role": "system", "content": "sys clean"},
            {"role": "user", "content": "old clean"},
            {"role": "user", "content": "new clean"},
        ]
        out = _apply_redaction((), kwargs, redacted)
        assert out["preamble"] == "sys clean"
        assert out["chat_history"][0]["message"] == "old clean"
        assert out["message"] == "new clean"

    def test_v1_preamble_only_redacted(self):
        from promptguard.patches.cohere_patch import _apply_redaction

        kwargs = {"preamble": "sys leak"}
        out = _apply_redaction((), kwargs, [{"role": "system", "content": "sys clean"}])
        assert out == {"preamble": "sys clean"}


# ── OpenAI / Anthropic object-message redaction ───────────────────────────


class TestOpenAIObjectRedaction:
    def test_object_message_rewritten(self):
        from promptguard.patches.openai_patch import _apply_redaction

        kwargs = {"messages": [_V2Message("user", "leak")]}
        out = _apply_redaction((), kwargs, [{"role": "user", "content": "clean"}])
        assert out["messages"][0].content == "clean"

    def test_unrewritable_object_returns_none(self):
        from promptguard.patches.openai_patch import _apply_redaction

        kwargs = {"messages": [_FrozenMessage("user", "leak")]}
        assert _apply_redaction((), kwargs, [{"role": "user", "content": "clean"}]) is None

    def test_unextracted_entry_does_not_consume_index(self):
        """Entries skipped by extraction (no role/content) must not shift
        the redaction indices for the following messages."""
        from promptguard.patches.openai_patch import _apply_redaction, _messages_to_guard_format

        skipped = object()
        messages = [skipped, {"role": "user", "content": "leak"}]
        assert len(_messages_to_guard_format(messages)) == 1

        out = _apply_redaction((), {"messages": messages}, [{"role": "user", "content": "clean"}])
        assert out["messages"][0] is skipped
        assert out["messages"][1]["content"] == "clean"

    def test_multimodal_content_rebuilt_as_text_part(self):
        from promptguard.patches.openai_patch import _apply_redaction

        messages = [{"role": "user", "content": [{"type": "text", "text": "leak"}]}]
        out = _apply_redaction((), {"messages": messages}, [{"role": "user", "content": "clean"}])
        assert out["messages"][0]["content"] == [{"type": "text", "text": "clean"}]

    def test_missing_counterpart_returns_none(self):
        from promptguard.patches.openai_patch import _apply_redaction

        messages = [
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ]
        assert (
            _apply_redaction((), {"messages": messages}, [{"role": "user", "content": "clean"}])
            is None
        )


class TestAnthropicObjectRedaction:
    def test_object_message_rewritten(self):
        from promptguard.patches.anthropic_patch import _apply_redaction

        kwargs = {"messages": [_V2Message("user", "leak")]}
        out = _apply_redaction((), kwargs, [{"role": "user", "content": "clean"}])
        assert out["messages"][0].content == "clean"

    def test_unrewritable_object_returns_none(self):
        from promptguard.patches.anthropic_patch import _apply_redaction

        kwargs = {"messages": [_FrozenMessage("user", "leak")]}
        assert _apply_redaction((), kwargs, [{"role": "user", "content": "clean"}]) is None

    def test_system_offset_preserved_with_objects(self):
        from promptguard.patches.anthropic_patch import _apply_redaction

        kwargs = {"system": "sys leak", "messages": [_V2Message("user", "user leak")]}
        redacted = [
            {"role": "system", "content": "sys clean"},
            {"role": "user", "content": "user clean"},
        ]
        out = _apply_redaction((), kwargs, redacted)
        assert out["system"] == "sys clean"
        assert out["messages"][0].content == "user clean"

    def test_missing_counterpart_returns_none(self):
        from promptguard.patches.anthropic_patch import _apply_redaction

        kwargs = {
            "system": "sys",
            "messages": [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}],
        }
        redacted = [
            {"role": "system", "content": "sys clean"},
            {"role": "user", "content": "one clean"},
        ]
        assert _apply_redaction((), kwargs, redacted) is None
