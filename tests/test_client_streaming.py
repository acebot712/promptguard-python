"""
Tests for streaming error handling and the retry/error helpers added in the
audit round: streamed 4xx/5xx must raise instead of yielding an empty iterator,
Retry-After is honored, quota-exhaustion 429s are not retried, and malformed
error bodies don't mask the real HTTP error.
"""

import httpx
import pytest

from promptguard.client import (
    PromptGuard,
    PromptGuardAsync,
    PromptGuardError,
    _is_non_retryable_error,
    _parse_error,
    _retry_after_seconds,
)

_ERR_BODY = b'{"error": {"message": "boom", "code": "BAD_REQUEST"}}'


class _FakeStreamResponse:
    def __init__(self, status_code: int, content: bytes = _ERR_BODY, headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}

    def read(self):
        return self.content

    async def aread(self):
        return self.content

    def json(self):
        import json

        return json.loads(self.content)

    def iter_lines(self):
        return iter([])

    async def aiter_lines(self):
        return
        yield  # pragma: no cover - makes this an async generator


class _FakeStreamCM:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self._response

    def __exit__(self, *exc):
        return False

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


# ── Streaming status handling ─────────────────────────────────────────────


class TestSyncStreamErrors:
    def test_stream_4xx_raises(self, monkeypatch):
        pg = PromptGuard(api_key="pg_test")
        monkeypatch.setattr(
            pg._http, "stream", lambda *a, **k: _FakeStreamCM(_FakeStreamResponse(400))
        )
        stream = pg.chat.completions.create(
            model="gpt", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        with pytest.raises(PromptGuardError) as exc:
            list(stream)
        assert exc.value.status_code == 400

    def test_stream_5xx_raises(self, monkeypatch):
        pg = PromptGuard(api_key="pg_test")
        monkeypatch.setattr(
            pg._http, "stream", lambda *a, **k: _FakeStreamCM(_FakeStreamResponse(503))
        )
        stream = pg.chat.completions.create(
            model="gpt", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        with pytest.raises(PromptGuardError):
            list(stream)


class _FakeOkStream:
    """A 200 stream that yields the given raw SSE lines."""

    def __init__(self, lines):
        self.status_code = 200
        self.content = b""
        self.headers = {}
        self._lines = lines

    def read(self):
        return self.content

    def json(self):
        return {}

    def iter_lines(self):
        return iter(self._lines)


class TestMalformedSSE:
    def test_malformed_data_line_raises_typed_error(self, monkeypatch):
        pg = PromptGuard(api_key="pg_test")
        monkeypatch.setattr(
            pg._http,
            "stream",
            lambda *a, **k: _FakeStreamCM(_FakeOkStream(["data: {not json}"])),
        )
        stream = pg.chat.completions.create(
            model="gpt", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        with pytest.raises(PromptGuardError) as exc:
            list(stream)
        assert exc.value.code == "INVALID_STREAM_DATA"

    def test_valid_data_lines_parsed(self, monkeypatch):
        pg = PromptGuard(api_key="pg_test")
        monkeypatch.setattr(
            pg._http,
            "stream",
            lambda *a, **k: _FakeStreamCM(_FakeOkStream(['data: {"chunk": 1}', "data: [DONE]"])),
        )
        stream = pg.chat.completions.create(
            model="gpt", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        assert list(stream) == [{"chunk": 1}]


class TestAsyncStreamErrors:
    @pytest.mark.asyncio
    async def test_stream_4xx_raises(self, monkeypatch):
        pg = PromptGuardAsync(api_key="pg_test")
        monkeypatch.setattr(
            pg._http, "stream", lambda *a, **k: _FakeStreamCM(_FakeStreamResponse(429))
        )
        stream = await pg.chat.completions.create(
            model="gpt", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        with pytest.raises(PromptGuardError) as exc:
            async for _ in stream:
                pass
        assert exc.value.status_code == 429
        await pg.close()


# ── Error/retry helpers ───────────────────────────────────────────────────


def _resp(status=429, json_body=None, headers=None, content=True):
    body = json_body if json_body is not None else {}
    return httpx.Response(
        status_code=status,
        json=body if content else None,
        headers=headers or {},
        request=httpx.Request("POST", "http://test"),
    )


class TestParseErrorGuards:
    def test_non_dict_body_does_not_crash(self):
        resp = httpx.Response(
            status_code=500,
            json=["unexpected", "array"],
            request=httpx.Request("POST", "http://test"),
        )
        err = _parse_error(resp)
        assert err.status_code == 500
        assert err.code == "UNKNOWN"

    def test_string_error_field(self):
        resp = httpx.Response(
            status_code=400,
            json={"error": "just a string"},
            request=httpx.Request("POST", "http://test"),
        )
        err = _parse_error(resp)
        assert err.code == "UNKNOWN"


class TestNonRetryable:
    def test_quota_exceeded_by_code(self):
        resp = _resp(429, {"error": {"code": "monthly_quota_exceeded"}})
        assert _is_non_retryable_error(resp) is True

    def test_quota_exceeded_by_type(self):
        resp = _resp(429, {"error": {"type": "monthly_quota_exceeded"}})
        assert _is_non_retryable_error(resp) is True

    def test_ordinary_429_is_retryable(self):
        resp = _resp(429, {"error": {"code": "rate_limited"}})
        assert _is_non_retryable_error(resp) is False


class TestRetryAfter:
    def test_missing_header(self):
        assert _retry_after_seconds(_resp(429)) is None

    def test_integer_seconds(self):
        assert _retry_after_seconds(_resp(429, headers={"Retry-After": "12"})) == 12.0

    def test_invalid_value(self):
        assert _retry_after_seconds(_resp(429, headers={"Retry-After": "soon"})) is None

    def test_http_date_is_clamped(self):
        # A far-future date must be clamped to the ceiling, not honored literally.
        resp = _resp(429, headers={"Retry-After": "Wed, 21 Oct 2099 07:28:00 GMT"})
        delay = _retry_after_seconds(resp)
        assert delay == 60.0

    def test_absurd_seconds_are_clamped(self):
        assert _retry_after_seconds(_resp(429, headers={"Retry-After": "1e300"})) == 60.0

    def test_non_finite_seconds(self):
        # inf clamps to the ceiling; NaN is treated as absent.
        assert _retry_after_seconds(_resp(429, headers={"Retry-After": "inf"})) == 60.0
        assert _retry_after_seconds(_resp(429, headers={"Retry-After": "nan"})) is None


class TestQuotaNotRetried:
    def test_quota_429_raises_without_retry(self, monkeypatch):
        pg = PromptGuard(api_key="pg_test", max_retries=3, retry_delay=0)
        calls = 0

        def fake_request(method, url, headers=None, **kwargs):
            nonlocal calls
            calls += 1
            return httpx.Response(
                status_code=429,
                json={"error": {"code": "monthly_quota_exceeded", "message": "no credits"}},
                request=httpx.Request(method, url),
            )

        monkeypatch.setattr(pg._http, "request", fake_request)
        with pytest.raises(PromptGuardError):
            pg.chat.completions.create(model="gpt", messages=[{"role": "user", "content": "hi"}])
        assert calls == 1  # never retried
