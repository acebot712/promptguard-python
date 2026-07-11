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


# ── Cohere redaction ──────────────────────────────────────────────────────


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
