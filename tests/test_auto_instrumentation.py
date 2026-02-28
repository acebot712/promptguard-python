"""
Tests for the auto-instrumentation system (init/shutdown and patching).
"""

from unittest.mock import patch

import pytest

from promptguard.auto import (
    get_guard_client,
    get_mode,
    init,
    is_fail_open,
    is_initialized,
    should_scan_responses,
    shutdown,
)


class TestInit:
    """Test the init() function."""

    def teardown_method(self):
        """Clean up after each test."""
        try:
            shutdown()
        except Exception:
            pass

    def test_init_requires_api_key(self):
        with pytest.raises(ValueError, match="API key required"):
            init(api_key="")

    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("PROMPTGUARD_API_KEY", "pg_test_env")
        with patch("promptguard.auto._apply_patches"):
            init()
        assert is_initialized()
        assert get_guard_client() is not None

    def test_init_explicit_key(self):
        with patch("promptguard.auto._apply_patches"):
            init(api_key="pg_test_explicit")
        assert is_initialized()
        assert get_guard_client() is not None

    def test_init_default_mode_is_enforce(self):
        with patch("promptguard.auto._apply_patches"):
            init(api_key="pg_test")
        assert get_mode() == "enforce"

    def test_init_monitor_mode(self):
        with patch("promptguard.auto._apply_patches"):
            init(api_key="pg_test", mode="monitor")
        assert get_mode() == "monitor"

    def test_init_invalid_mode(self):
        with pytest.raises(ValueError, match="mode must be"):
            init(api_key="pg_test", mode="invalid")

    def test_init_fail_open_default(self):
        with patch("promptguard.auto._apply_patches"):
            init(api_key="pg_test")
        assert is_fail_open()

    def test_init_fail_closed(self):
        with patch("promptguard.auto._apply_patches"):
            init(api_key="pg_test", fail_open=False)
        assert not is_fail_open()

    def test_init_scan_responses_default_off(self):
        with patch("promptguard.auto._apply_patches"):
            init(api_key="pg_test")
        assert not should_scan_responses()

    def test_init_scan_responses_on(self):
        with patch("promptguard.auto._apply_patches"):
            init(api_key="pg_test", scan_responses=True)
        assert should_scan_responses()


class TestShutdown:
    """Test the shutdown() function."""

    def test_shutdown_clears_state(self):
        with patch("promptguard.auto._apply_patches"):
            init(api_key="pg_test")
        assert is_initialized()

        with patch("promptguard.auto._remove_patches"):
            shutdown()
        assert not is_initialized()
        assert get_guard_client() is None

    def test_shutdown_idempotent(self):
        """Calling shutdown without init should not error."""
        with patch("promptguard.auto._remove_patches"):
            shutdown()
        assert not is_initialized()


class TestOpenAIPatch:
    """Test the OpenAI SDK patch logic."""

    def test_messages_to_guard_format_dicts(self):
        from promptguard.patches.openai_patch import _messages_to_guard_format

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        result = _messages_to_guard_format(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."
        assert result[1]["role"] == "user"

    def test_messages_to_guard_format_multimodal(self):
        from promptguard.patches.openai_patch import _messages_to_guard_format

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            }
        ]
        result = _messages_to_guard_format(messages)
        assert len(result) == 1
        assert result[0]["content"] == "What is in this image?"

    def test_messages_to_guard_format_empty(self):
        from promptguard.patches.openai_patch import _messages_to_guard_format

        assert _messages_to_guard_format(None) == []
        assert _messages_to_guard_format([]) == []

    def test_apply_redaction(self):
        from promptguard.patches.openai_patch import _apply_redaction

        messages = [
            {"role": "user", "content": "My SSN is 123-45-6789"},
        ]
        redacted = [
            {"role": "user", "content": "My SSN is [REDACTED]"},
        ]
        result_kwargs = _apply_redaction((), {"messages": messages}, redacted)
        assert result_kwargs["messages"][0]["content"] == "My SSN is [REDACTED]"


class TestAnthropicPatch:
    """Test the Anthropic SDK patch logic."""

    def test_messages_with_system(self):
        from promptguard.patches.anthropic_patch import _messages_to_guard_format

        messages = [{"role": "user", "content": "Hello"}]
        result = _messages_to_guard_format(messages, system="Be helpful")
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be helpful"
        assert result[1]["role"] == "user"

    def test_messages_without_system(self):
        from promptguard.patches.anthropic_patch import _messages_to_guard_format

        messages = [{"role": "user", "content": "Hello"}]
        result = _messages_to_guard_format(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_system_as_content_blocks(self):
        from promptguard.patches.anthropic_patch import _messages_to_guard_format

        system = [{"type": "text", "text": "You are a helper."}]
        messages = [{"role": "user", "content": "Hi"}]
        result = _messages_to_guard_format(messages, system=system)
        assert result[0]["content"] == "You are a helper."


class TestGooglePatch:
    """Test the Google Generative AI patch logic."""

    def test_string_content(self):
        from promptguard.patches.google_patch import _content_to_guard_format

        result = _content_to_guard_format("What is AI?")
        assert len(result) == 1
        assert result[0]["content"] == "What is AI?"

    def test_list_of_strings(self):
        from promptguard.patches.google_patch import _content_to_guard_format

        result = _content_to_guard_format(["Hello", "World"])
        assert len(result) == 2

    def test_dict_with_parts(self):
        from promptguard.patches.google_patch import _content_to_guard_format

        result = _content_to_guard_format(
            [
                {"role": "user", "parts": [{"text": "Hello"}]},
            ]
        )
        assert len(result) == 1
        assert result[0]["content"] == "Hello"


class TestBedrockPatch:
    """Test the AWS Bedrock patch logic."""

    def test_anthropic_format(self):
        from promptguard.patches.bedrock_patch import _extract_messages_from_body

        body = {
            "messages": [
                {"role": "user", "content": "Hello Claude"},
            ],
            "system": "Be helpful",
        }
        result = _extract_messages_from_body(body)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "Hello Claude"

    def test_titan_format(self):
        from promptguard.patches.bedrock_patch import _extract_messages_from_body

        body = {"inputText": "What is machine learning?"}
        result = _extract_messages_from_body(body)
        assert len(result) == 1
        assert result[0]["content"] == "What is machine learning?"

    def test_llama_format(self):
        from promptguard.patches.bedrock_patch import _extract_messages_from_body

        body = {"prompt": "Tell me about Python."}
        result = _extract_messages_from_body(body)
        assert len(result) == 1
        assert result[0]["content"] == "Tell me about Python."

    def test_bytes_body(self):
        import json

        from promptguard.patches.bedrock_patch import _extract_messages_from_body

        body = json.dumps({"inputText": "Hello from bytes"}).encode()
        result = _extract_messages_from_body(body)
        assert len(result) == 1
        assert result[0]["content"] == "Hello from bytes"

    def test_converse_api_format(self):
        from promptguard.patches.bedrock_patch import _extract_messages_from_body

        body = {
            "Messages": [
                {"role": "user", "content": [{"text": "Hello"}]},
            ],
            "System": [{"text": "Be helpful"}],
        }
        result = _extract_messages_from_body(body)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "Hello"


class TestCoherePatch:
    """Test the Cohere SDK patch logic."""

    def test_v1_message_format(self):
        from promptguard.patches.cohere_patch import _to_guard_messages

        result = _to_guard_messages(
            message="Hello",
            chat_history=[
                {"role": "user", "message": "Previous"},
                {"role": "chatbot", "message": "Response"},
            ],
        )
        assert len(result) == 3
        assert result[2]["content"] == "Hello"

    def test_v2_messages_format(self):
        from promptguard.patches.cohere_patch import _to_guard_messages

        result = _to_guard_messages(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        )
        assert len(result) == 2
        assert result[0]["content"] == "Hello"
