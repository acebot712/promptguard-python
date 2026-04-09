"""
PromptGuard Client - Drop-in replacement for OpenAI client.

Provides the same interface as OpenAI's client but routes through PromptGuard
for security scanning, caching, and cost optimization.
"""

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from promptguard._version import __version__
from promptguard.config import Config

_SDK_LANG = "python"
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class PromptGuardError(Exception):
    """Error from PromptGuard API"""

    def __init__(
        self,
        message: str,
        code: str,
        status_code: int,
        *,
        error_type: str | None = None,
        upgrade_url: str | None = None,
        current_plan: str | None = None,
        requests_used: int | None = None,
        requests_limit: int | None = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.error_type = error_type
        self.upgrade_url = upgrade_url
        self.current_plan = current_plan
        self.requests_used = requests_used
        self.requests_limit = requests_limit
        super().__init__(f"{code}: {message}")


# -- Shared helpers used by both sync and async clients ----------------------


def _sdk_headers(api_key: str) -> dict[str, str]:
    return {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
        "X-PromptGuard-SDK": _SDK_LANG,
        "X-PromptGuard-Version": __version__,
    }


def _parse_error(response: httpx.Response) -> PromptGuardError:
    try:
        data = response.json() if response.content else {}
    except (json.JSONDecodeError, ValueError):
        data = {}
    err = data.get("error", {})
    return PromptGuardError(
        message=err.get("message", "Request failed"),
        code=err.get("code", "UNKNOWN"),
        status_code=response.status_code,
        error_type=err.get("type"),
        upgrade_url=err.get("upgrade_url"),
        current_plan=err.get("current_plan"),
        requests_used=err.get("requests_used"),
        requests_limit=err.get("requests_limit"),
    )


def _init_config(
    api_key: str | None,
    base_url: str | None,
    config: Config | None,
) -> Config:
    if config:
        return config
    return Config(
        api_key=api_key or os.environ.get("PROMPTGUARD_API_KEY", ""),
        base_url=base_url
        or os.environ.get("PROMPTGUARD_BASE_URL", "https://api.promptguard.co/api/v1/proxy"),
    )


# ── Sync Namespace Classes ─────────────────────────────────────────────────


class ChatCompletions:
    def __init__(self, client: "PromptGuard"):
        self._client = client

    def create(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stream:
            return self._stream(payload)
        return self._client._request("POST", "/chat/completions", json=payload)

    def _stream(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        payload["stream"] = True
        with self._client._http.stream(
            "POST",
            f"{self._client.config.base_url}/chat/completions",
            json=payload,
            headers=_sdk_headers(self._client.config.api_key),
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    yield json.loads(data)


class Chat:
    def __init__(self, client: "PromptGuard"):
        self.completions = ChatCompletions(client)


class Completions:
    """Legacy completions API (deprecated by OpenAI)."""

    def __init__(self, client: "PromptGuard"):
        self._client = client

    def create(
        self,
        model: str,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        **kwargs,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            **kwargs,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return self._client._request("POST", "/completions", json=payload)


class Embeddings:
    def __init__(self, client: "PromptGuard"):
        self._client = client

    def create(
        self,
        model: str,
        input: str | list[str],
        **kwargs,
    ) -> dict[str, Any]:
        return self._client._request(
            "POST",
            "/embeddings",
            json={"model": model, "input": input, **kwargs},
        )


class Security:
    def __init__(self, client: "PromptGuard"):
        self._client = client

    def scan(self, content: str, content_type: str = "prompt") -> dict[str, Any]:
        return self._client._request(
            "POST",
            "/security/scan",
            json={"content": content, "type": content_type},
        )

    def redact(self, content: str, pii_types: list[str] | None = None) -> dict[str, Any]:
        return self._client._request(
            "POST",
            "/security/redact",
            json={"content": content, "pii_types": pii_types},
        )


class Scrape:
    def __init__(self, client: "PromptGuard"):
        self._client = client

    def url(
        self,
        url: str,
        render_js: bool = False,
        extract_text: bool = True,
        timeout: int = 30,
    ) -> dict[str, Any]:
        return self._client._request(
            "POST",
            "/scrape",
            json={
                "url": url,
                "render_js": render_js,
                "extract_text": extract_text,
                "timeout": timeout,
            },
        )

    def batch(self, urls: list[str], **kwargs) -> dict[str, Any]:
        return self._client._request(
            "POST",
            "/scrape/batch",
            json={"urls": urls, **kwargs},
        )


class Agent:
    def __init__(self, client: "PromptGuard"):
        self._client = client

    def validate_tool(
        self,
        agent_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        return self._client._request(
            "POST",
            "/agent/validate-tool",
            json={
                "agent_id": agent_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "session_id": session_id,
            },
        )

    def stats(self, agent_id: str) -> dict[str, Any]:
        return self._client._request("GET", f"/agent/{agent_id}/stats")


class RedTeam:
    _BASE = "/internal/redteam"

    def __init__(self, client: "PromptGuard"):
        self._client = client

    def list_tests(self) -> dict[str, Any]:
        return self._client._request("GET", f"{self._BASE}/tests")

    def run_test(self, test_name: str, target_preset: str = "default") -> dict[str, Any]:
        return self._client._request(
            "POST",
            f"{self._BASE}/test/{test_name}",
            json={"target_preset": target_preset},
        )

    def run_all(self, target_preset: str = "default") -> dict[str, Any]:
        return self._client._request(
            "POST",
            f"{self._BASE}/test-all",
            json={"target_preset": target_preset},
        )

    def run_custom(self, prompt: str, target_preset: str = "default") -> dict[str, Any]:
        return self._client._request(
            "POST",
            f"{self._BASE}/test-custom",
            json={"custom_prompt": prompt, "target_preset": target_preset},
        )

    def run_autonomous(
        self,
        budget: int = 100,
        target_preset: str = "default",
        enabled_detectors: list[str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"budget": budget, "target_preset": target_preset}
        if enabled_detectors is not None:
            payload["enabled_detectors"] = enabled_detectors
        return self._client._request("POST", f"{self._BASE}/autonomous", json=payload)

    def intelligence_stats(self) -> dict[str, Any]:
        return self._client._request("GET", f"{self._BASE}/intelligence/stats")


# ── Sync Client ────────────────────────────────────────────────────────────


class PromptGuard:
    """
    PromptGuard client - Drop-in security for AI applications.

    Usage::

        from promptguard import PromptGuard

        pg = PromptGuard(api_key="pg_xxx")
        response = pg.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        scan_result = pg.security.scan("Check this content")
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        config: Config | None = None,
        timeout: float = 30.0,
    ):
        self.config = _init_config(api_key, base_url, config)
        if not self.config.api_key:
            raise ValueError(
                "API key required. Pass api_key parameter or set "
                "PROMPTGUARD_API_KEY environment variable."
            )
        self._http = httpx.Client(timeout=timeout)
        self.chat = Chat(self)
        self.completions = Completions(self)
        self.embeddings = Embeddings(self)
        self.security = Security(self)
        self.scrape = Scrape(self)
        self.agent = Agent(self)
        self.redteam = RedTeam(self)

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        headers = _sdk_headers(self.config.api_key)
        last_exc: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._http.request(method, url, headers=headers, **kwargs)
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay * (2**attempt))
                    continue
                raise

            if (
                response.status_code in _RETRYABLE_STATUS_CODES
                and attempt < self.config.max_retries
            ):
                time.sleep(self.config.retry_delay * (2**attempt))
                continue

            if response.status_code >= 400:
                raise _parse_error(response)

            return response.json()

        if last_exc:
            raise last_exc
        raise PromptGuardError(message="Max retries exceeded", code="MAX_RETRIES", status_code=0)

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Async Namespace Classes ────────────────────────────────────────────────
#
# Mirror the sync classes above with async/await.  This duplication is an
# inherent Python limitation: sync and async callables have different types.


class AsyncChatCompletions:
    def __init__(self, client: "PromptGuardAsync"):
        self._client = client

    async def create(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stream:
            return self._stream(payload)
        return await self._client._request("POST", "/chat/completions", json=payload)

    async def _stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        payload["stream"] = True
        async with self._client._http.stream(
            "POST",
            f"{self._client.config.base_url}/chat/completions",
            json=payload,
            headers=_sdk_headers(self._client.config.api_key),
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    yield json.loads(data)


class AsyncChat:
    def __init__(self, client: "PromptGuardAsync"):
        self.completions = AsyncChatCompletions(client)


class AsyncCompletions:
    def __init__(self, client: "PromptGuardAsync"):
        self._client = client

    async def create(
        self,
        model: str,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        **kwargs,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            **kwargs,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return await self._client._request("POST", "/completions", json=payload)


class AsyncEmbeddings:
    def __init__(self, client: "PromptGuardAsync"):
        self._client = client

    async def create(
        self,
        model: str,
        input: str | list[str],
        **kwargs,
    ) -> dict[str, Any]:
        return await self._client._request(
            "POST",
            "/embeddings",
            json={"model": model, "input": input, **kwargs},
        )


class AsyncSecurity:
    def __init__(self, client: "PromptGuardAsync"):
        self._client = client

    async def scan(self, content: str, content_type: str = "prompt") -> dict[str, Any]:
        return await self._client._request(
            "POST",
            "/security/scan",
            json={"content": content, "type": content_type},
        )

    async def redact(self, content: str, pii_types: list[str] | None = None) -> dict[str, Any]:
        return await self._client._request(
            "POST",
            "/security/redact",
            json={"content": content, "pii_types": pii_types},
        )


class AsyncScrape:
    def __init__(self, client: "PromptGuardAsync"):
        self._client = client

    async def url(
        self,
        url: str,
        render_js: bool = False,
        extract_text: bool = True,
        timeout: int = 30,
    ) -> dict[str, Any]:
        return await self._client._request(
            "POST",
            "/scrape",
            json={
                "url": url,
                "render_js": render_js,
                "extract_text": extract_text,
                "timeout": timeout,
            },
        )

    async def batch(self, urls: list[str], **kwargs) -> dict[str, Any]:
        return await self._client._request(
            "POST",
            "/scrape/batch",
            json={"urls": urls, **kwargs},
        )


class AsyncAgent:
    def __init__(self, client: "PromptGuardAsync"):
        self._client = client

    async def validate_tool(
        self,
        agent_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._client._request(
            "POST",
            "/agent/validate-tool",
            json={
                "agent_id": agent_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "session_id": session_id,
            },
        )

    async def stats(self, agent_id: str) -> dict[str, Any]:
        return await self._client._request("GET", f"/agent/{agent_id}/stats")


class AsyncRedTeam:
    _BASE = "/internal/redteam"

    def __init__(self, client: "PromptGuardAsync"):
        self._client = client

    async def list_tests(self) -> dict[str, Any]:
        return await self._client._request("GET", f"{self._BASE}/tests")

    async def run_test(self, test_name: str, target_preset: str = "default") -> dict[str, Any]:
        return await self._client._request(
            "POST",
            f"{self._BASE}/test/{test_name}",
            json={"target_preset": target_preset},
        )

    async def run_all(self, target_preset: str = "default") -> dict[str, Any]:
        return await self._client._request(
            "POST",
            f"{self._BASE}/test-all",
            json={"target_preset": target_preset},
        )

    async def run_custom(self, prompt: str, target_preset: str = "default") -> dict[str, Any]:
        return await self._client._request(
            "POST",
            f"{self._BASE}/test-custom",
            json={"custom_prompt": prompt, "target_preset": target_preset},
        )

    async def run_autonomous(
        self,
        budget: int = 100,
        target_preset: str = "default",
        enabled_detectors: list[str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"budget": budget, "target_preset": target_preset}
        if enabled_detectors is not None:
            payload["enabled_detectors"] = enabled_detectors
        return await self._client._request("POST", f"{self._BASE}/autonomous", json=payload)

    async def intelligence_stats(self) -> dict[str, Any]:
        return await self._client._request("GET", f"{self._BASE}/intelligence/stats")


# ── Async Client ───────────────────────────────────────────────────────────


class PromptGuardAsync:
    """
    Async PromptGuard client -- full parity with the sync client.

    Usage::

        async with PromptGuardAsync(api_key="pg_xxx") as pg:
            resp = await pg.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": "Hi"}],
            )
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        config: Config | None = None,
        timeout: float = 30.0,
    ):
        self.config = _init_config(api_key, base_url, config)
        if not self.config.api_key:
            raise ValueError(
                "API key required. Pass api_key or set PROMPTGUARD_API_KEY environment variable."
            )
        self._http = httpx.AsyncClient(timeout=timeout)
        self.chat = AsyncChat(self)
        self.completions = AsyncCompletions(self)
        self.embeddings = AsyncEmbeddings(self)
        self.security = AsyncSecurity(self)
        self.scrape = AsyncScrape(self)
        self.agent = AsyncAgent(self)
        self.redteam = AsyncRedTeam(self)

    async def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        headers = _sdk_headers(self.config.api_key)
        last_exc: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self._http.request(method, url, headers=headers, **kwargs)
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * (2**attempt))
                    continue
                raise

            if (
                response.status_code in _RETRYABLE_STATUS_CODES
                and attempt < self.config.max_retries
            ):
                await asyncio.sleep(self.config.retry_delay * (2**attempt))
                continue

            if response.status_code >= 400:
                raise _parse_error(response)

            return response.json()

        if last_exc:
            raise last_exc
        raise PromptGuardError(message="Max retries exceeded", code="MAX_RETRIES", status_code=0)

    async def close(self):
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
