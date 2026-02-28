"""
Tests for PromptGuard and PromptGuardAsync clients.

Covers:
- Sync/async API namespace parity
- Retry/backoff logic
- SDK version headers
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from promptguard.client import (
    _SDK_LANG,
    _SDK_VERSION,
    PromptGuard,
    PromptGuardAsync,
    PromptGuardError,
)
from promptguard.config import Config

# ── Helpers ────────────────────────────────────────────────────────────


def _ok_response(data=None, status=200):
    return httpx.Response(
        status_code=status,
        json=data or {"ok": True},
        request=httpx.Request("POST", "http://test"),
    )


def _err_response(status=500, msg="fail", code="ERR"):
    return httpx.Response(
        status_code=status,
        json={"error": {"message": msg, "code": code}},
        request=httpx.Request("POST", "http://test"),
    )


def _make_sync_client(**kw):
    return PromptGuard(api_key="pg_test", **kw)


def _make_async_client(**kw):
    return PromptGuardAsync(api_key="pg_test", **kw)


# ── Sync client namespace parity ──────────────────────────────────────


class TestSyncClientNamespaces:
    def test_has_all_namespaces(self):
        c = _make_sync_client()
        assert hasattr(c, "chat")
        assert hasattr(c.chat, "completions")
        assert hasattr(c, "completions")
        assert hasattr(c, "embeddings")
        assert hasattr(c, "security")
        assert hasattr(c, "scrape")
        assert hasattr(c, "agent")
        assert hasattr(c, "redteam")

    def test_chat_completions_create(self):
        c = _make_sync_client()
        assert callable(c.chat.completions.create)

    def test_embeddings_create(self):
        c = _make_sync_client()
        assert callable(c.embeddings.create)

    def test_completions_create(self):
        c = _make_sync_client()
        assert callable(c.completions.create)

    def test_security_scan_and_redact(self):
        c = _make_sync_client()
        assert callable(c.security.scan)
        assert callable(c.security.redact)

    def test_scrape_url_and_batch(self):
        c = _make_sync_client()
        assert callable(c.scrape.url)
        assert callable(c.scrape.batch)

    def test_agent_validate_tool_and_stats(self):
        c = _make_sync_client()
        assert callable(c.agent.validate_tool)
        assert callable(c.agent.stats)

    def test_redteam_methods(self):
        c = _make_sync_client()
        assert callable(c.redteam.list_tests)
        assert callable(c.redteam.run_test)
        assert callable(c.redteam.run_all)
        assert callable(c.redteam.run_custom)


# ── Async client namespace parity ─────────────────────────────────────


class TestAsyncClientNamespaces:
    def test_has_all_namespaces(self):
        c = _make_async_client()
        assert hasattr(c, "chat")
        assert hasattr(c.chat, "completions")
        assert hasattr(c, "completions")
        assert hasattr(c, "embeddings")
        assert hasattr(c, "security")
        assert hasattr(c, "scrape")
        assert hasattr(c, "agent")
        assert hasattr(c, "redteam")

    def test_chat_completions_create_is_coroutine(self):
        c = _make_async_client()
        assert asyncio.iscoroutinefunction(c.chat.completions.create)

    def test_completions_create_is_coroutine(self):
        c = _make_async_client()
        assert asyncio.iscoroutinefunction(c.completions.create)

    def test_embeddings_create_is_coroutine(self):
        c = _make_async_client()
        assert asyncio.iscoroutinefunction(c.embeddings.create)

    def test_security_scan_is_coroutine(self):
        c = _make_async_client()
        assert asyncio.iscoroutinefunction(c.security.scan)

    def test_security_redact_is_coroutine(self):
        c = _make_async_client()
        assert asyncio.iscoroutinefunction(c.security.redact)

    def test_scrape_url_is_coroutine(self):
        c = _make_async_client()
        assert asyncio.iscoroutinefunction(c.scrape.url)

    def test_agent_validate_tool_is_coroutine(self):
        c = _make_async_client()
        assert asyncio.iscoroutinefunction(c.agent.validate_tool)

    def test_redteam_run_test_is_coroutine(self):
        c = _make_async_client()
        assert asyncio.iscoroutinefunction(c.redteam.run_test)


# ── SDK headers ───────────────────────────────────────────────────────


class TestSDKHeaders:
    def test_sync_headers(self):
        c = _make_sync_client()
        h = c._get_headers()
        assert h["X-PromptGuard-SDK"] == _SDK_LANG
        assert h["X-PromptGuard-Version"] == _SDK_VERSION
        assert "Bearer pg_test" in h["Authorization"]

    def test_async_headers(self):
        c = _make_async_client()
        h = c._get_headers()
        assert h["X-PromptGuard-SDK"] == _SDK_LANG
        assert h["X-PromptGuard-Version"] == _SDK_VERSION
        assert "Bearer pg_test" in h["Authorization"]


# ── Retry logic (sync) ───────────────────────────────────────────────


class TestSyncRetry:
    @patch("time.sleep")
    def test_retries_on_500(self, mock_sleep):
        c = _make_sync_client()
        c._client = MagicMock()
        c._client.request.side_effect = [
            _err_response(500),
            _err_response(500),
            _ok_response({"result": "ok"}),
        ]
        result = c._request("POST", "/test")
        assert result == {"result": "ok"}
        assert c._client.request.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("time.sleep")
    def test_retries_on_429(self, mock_sleep):
        c = _make_sync_client()
        c._client = MagicMock()
        c._client.request.side_effect = [
            _err_response(429),
            _ok_response(),
        ]
        result = c._request("POST", "/test")
        assert result == {"ok": True}
        assert mock_sleep.call_count == 1

    @patch("time.sleep")
    def test_no_retry_on_400(self, mock_sleep):
        c = _make_sync_client()
        c._client = MagicMock()
        c._client.request.return_value = _err_response(400)
        with pytest.raises(PromptGuardError) as exc:
            c._request("POST", "/test")
        assert exc.value.status_code == 400
        assert c._client.request.call_count == 1
        mock_sleep.assert_not_called()

    @patch("time.sleep")
    def test_exhausts_retries(self, mock_sleep):
        cfg = Config(api_key="pg_test", max_retries=2)
        c = PromptGuard(config=cfg)
        c._client = MagicMock()
        c._client.request.return_value = _err_response(503)
        with pytest.raises(PromptGuardError):
            c._request("POST", "/test")
        assert c._client.request.call_count == 3  # initial + 2 retries

    @patch("time.sleep")
    def test_retries_on_transport_error(self, mock_sleep):
        c = _make_sync_client()
        c._client = MagicMock()
        c._client.request.side_effect = [
            httpx.ConnectError("refused"),
            _ok_response({"recovered": True}),
        ]
        result = c._request("POST", "/test")
        assert result == {"recovered": True}

    @patch("time.sleep")
    def test_exponential_backoff(self, mock_sleep):
        cfg = Config(api_key="pg_test", max_retries=3, retry_delay=1.0)
        c = PromptGuard(config=cfg)
        c._client = MagicMock()
        c._client.request.side_effect = [
            _err_response(500),
            _err_response(500),
            _err_response(500),
            _ok_response(),
        ]
        c._request("POST", "/test")
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [1.0, 2.0, 4.0]


# ── Retry logic (async) ──────────────────────────────────────────────


class TestAsyncRetry:
    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_on_500(self, mock_sleep):
        c = _make_async_client()
        c._http = AsyncMock()
        c._http.request = AsyncMock(
            side_effect=[
                _err_response(500),
                _err_response(500),
                _ok_response({"result": "ok"}),
            ]
        )
        result = await c._request("POST", "/test")
        assert result == {"result": "ok"}
        assert c._http.request.call_count == 3
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_no_retry_on_400(self, mock_sleep):
        c = _make_async_client()
        c._http = AsyncMock()
        c._http.request = AsyncMock(return_value=_err_response(400))
        with pytest.raises(PromptGuardError) as exc:
            await c._request("POST", "/test")
        assert exc.value.status_code == 400
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_on_transport_error(self, mock_sleep):
        c = _make_async_client()
        c._http = AsyncMock()
        c._http.request = AsyncMock(
            side_effect=[
                httpx.ConnectError("refused"),
                _ok_response({"recovered": True}),
            ]
        )
        result = await c._request("POST", "/test")
        assert result == {"recovered": True}


# ── Context managers ──────────────────────────────────────────────────


class TestContextManagers:
    def test_sync_context_manager(self):
        with _make_sync_client() as c:
            assert c is not None

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        async with _make_async_client() as c:
            assert c is not None


# ── Validation ────────────────────────────────────────────────────────


class TestValidation:
    def test_sync_requires_api_key(self):
        with pytest.raises(ValueError, match="API key"):
            PromptGuard(api_key="")

    def test_async_requires_api_key(self):
        with pytest.raises(ValueError, match="API key"):
            PromptGuardAsync(api_key="")
