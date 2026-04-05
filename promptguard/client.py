"""
PromptGuard Client - Drop-in replacement for OpenAI client

Provides the same interface as OpenAI's client but routes through PromptGuard
for security scanning, caching, and cost optimization.
"""

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator, Iterator
from typing import (
    Any,
)

import httpx

from promptguard.config import Config

_SDK_LANG = "python"
_SDK_VERSION = "1.5.3"

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class ChatCompletions:
    """Chat completions API (OpenAI-compatible)"""

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
        """
        Create a chat completion.

        This is a drop-in replacement for openai.chat.completions.create()
        but routes through PromptGuard for security scanning.

        Args:
            model: Model name (e.g., "gpt-4", "claude-3-sonnet")
            messages: List of message dicts with role and content
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters passed to the model

        Returns:
            Completion response (same format as OpenAI)
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if stream:
            return self._stream_completion(payload)

        return self._client._request("POST", "/chat/completions", json=payload)

    def _stream_completion(
        self,
        payload: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        """Stream a chat completion"""
        payload["stream"] = True

        with self._client._client.stream(
            "POST",
            f"{self._client.config.base_url}/chat/completions",
            json=payload,
            headers=self._client._get_headers(),
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    yield json.loads(data)


class Chat:
    """Chat API namespace"""

    def __init__(self, client: "PromptGuard"):
        self.completions = ChatCompletions(client)


class Completions:
    """Legacy completions API (OpenAI-compatible)"""

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
        """Create a completion (legacy API)"""
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            **kwargs,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        return self._client._request("POST", "/completions", json=payload)


class Embeddings:
    """Embeddings API"""

    def __init__(self, client: "PromptGuard"):
        self._client = client

    def create(
        self,
        model: str,
        input: str | list[str],
        **kwargs,
    ) -> dict[str, Any]:
        """Create embeddings"""
        payload = {
            "model": model,
            "input": input,
            **kwargs,
        }

        return self._client._request("POST", "/embeddings", json=payload)


class Security:
    """PromptGuard-specific security APIs"""

    def __init__(self, client: "PromptGuard"):
        self._client = client

    def scan(
        self,
        content: str,
        content_type: str = "prompt",
    ) -> dict[str, Any]:
        """
        Scan content for security issues.

        Args:
            content: Text to scan
            content_type: "prompt" or "response"

        Returns:
            Scan result with threats detected
        """
        return self._client._request(
            "POST",
            "/security/scan",
            json={"content": content, "type": content_type},
        )

    def redact(
        self,
        content: str,
        pii_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Redact PII from content.

        Args:
            content: Text to redact
            pii_types: Specific PII types to redact (default: all)

        Returns:
            Redacted content
        """
        return self._client._request(
            "POST",
            "/security/redact",
            json={"content": content, "pii_types": pii_types},
        )


class Scrape:
    """PromptGuard Secure Web Scraping API"""

    def __init__(self, client: "PromptGuard"):
        self._client = client

    def url(
        self,
        url: str,
        render_js: bool = False,
        extract_text: bool = True,
        timeout: int = 30,
    ) -> dict[str, Any]:
        """
        Securely scrape a URL with threat scanning.

        Args:
            url: URL to scrape
            render_js: Whether to render JavaScript
            extract_text: Whether to extract clean text
            timeout: Request timeout in seconds

        Returns:
            Dict with content and threat analysis
        """
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

    def batch(
        self,
        urls: list[str],
        **kwargs,
    ) -> dict[str, Any]:
        """
        Batch scrape multiple URLs.

        Args:
            urls: List of URLs to scrape
            **kwargs: Same options as single scrape

        Returns:
            Dict with job_id for tracking
        """
        return self._client._request(
            "POST",
            "/scrape/batch",
            json={"urls": urls, **kwargs},
        )


class Agent:
    """PromptGuard AI Agent Security API"""

    def __init__(self, client: "PromptGuard"):
        self._client = client

    def validate_tool(
        self,
        agent_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Validate a tool call before execution.

        Args:
            agent_id: Unique identifier for the agent
            tool_name: Name of the tool being called
            arguments: Arguments being passed to the tool
            session_id: Optional session identifier

        Returns:
            Dict with allowed status, risk score, and warnings
        """
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
        """Get statistics for an agent"""
        return self._client._request("GET", f"/agent/{agent_id}/stats")


class RedTeam:
    """PromptGuard Red Team Testing API"""

    def __init__(self, client: "PromptGuard"):
        self._client = client
        # Note: RedTeam API requires internal/admin access
        self._base = "/internal/redteam"

    def list_tests(self) -> dict[str, Any]:
        """List all available red team tests"""
        return self._client._request("GET", f"{self._base}/tests")

    def run_test(
        self,
        test_name: str,
        target_preset: str = "default",
    ) -> dict[str, Any]:
        """
        Run a specific red team test.

        Args:
            test_name: Name of the test to run
            target_preset: Policy preset to test against

        Returns:
            Test result with decision and details
        """
        return self._client._request(
            "POST",
            f"{self._base}/test/{test_name}",
            json={"target_preset": target_preset},
        )

    def run_all(
        self,
        target_preset: str = "default",
    ) -> dict[str, Any]:
        """
        Run all red team tests.

        Args:
            target_preset: Policy preset to test against

        Returns:
            Summary with all test results
        """
        return self._client._request(
            "POST",
            f"{self._base}/test-all",
            json={"target_preset": target_preset},
        )

    def run_custom(
        self,
        prompt: str,
        target_preset: str = "default",
    ) -> dict[str, Any]:
        """
        Run a custom adversarial prompt test.

        Args:
            prompt: Custom adversarial prompt to test
            target_preset: Policy preset to test against

        Returns:
            Test result
        """
        return self._client._request(
            "POST",
            f"{self._base}/test-custom",
            json={
                "custom_prompt": prompt,
                "target_preset": target_preset,
            },
        )

    def run_autonomous(
        self,
        budget: int = 100,
        target_preset: str = "default",
        enabled_detectors: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run the autonomous red team agent.

        Uses LLM-powered mutation to discover novel attack
        vectors that bypass the current policy configuration.

        Args:
            budget: Max iterations (1-1000). Higher = more thorough.
            target_preset: Policy preset to test against.
            enabled_detectors: Limit to specific detectors.

        Returns:
            Report with grade, bypass rate, and discovered bypasses.
        """
        payload: dict[str, Any] = {
            "budget": budget,
            "target_preset": target_preset,
        }
        if enabled_detectors is not None:
            payload["enabled_detectors"] = enabled_detectors
        return self._client._request(
            "POST",
            f"{self._base}/autonomous",
            json=payload,
        )

    def intelligence_stats(self) -> dict[str, Any]:
        """Get statistics from the Attack Intelligence DB."""
        return self._client._request(
            "GET",
            f"{self._base}/intelligence/stats",
        )


class PromptGuard:
    """
    PromptGuard client - Drop-in security for AI applications.

    Usage:
        from promptguard import PromptGuard

        # Initialize client
        pg = PromptGuard(api_key="pg_xxx")

        # Use like OpenAI client
        response = pg.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )

        # Security-specific features
        scan_result = pg.security.scan("Check this content")
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        config: Config | None = None,
        timeout: float = 30.0,
    ):
        """
        Initialize PromptGuard client.

        Args:
            api_key: PromptGuard API key (or set PROMPTGUARD_API_KEY env var)
            base_url: API base URL (default: https://api.promptguard.co/api/v1/proxy)
            config: Optional Config object
            timeout: Request timeout in seconds
        """
        self.config = config or Config(
            api_key=api_key or os.environ.get("PROMPTGUARD_API_KEY", ""),
            base_url=base_url
            or os.environ.get("PROMPTGUARD_BASE_URL", "https://api.promptguard.co/api/v1/proxy"),
        )

        if not self.config.api_key:
            raise ValueError(
                "API key required. Pass api_key parameter or set "
                "PROMPTGUARD_API_KEY environment variable."
            )

        self._client = httpx.Client(timeout=timeout)

        # API namespaces (OpenAI-compatible)
        self.chat = Chat(self)
        self.completions = Completions(self)
        self.embeddings = Embeddings(self)

        # PromptGuard-specific APIs
        self.security = Security(self)
        self.scrape = Scrape(self)
        self.agent = Agent(self)
        self.redteam = RedTeam(self)

    def _get_headers(self) -> dict[str, str]:
        """Get request headers"""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "X-PromptGuard-SDK": _SDK_LANG,
            "X-PromptGuard-Version": _SDK_VERSION,
        }

    def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Make API request with retry/backoff."""
        url = f"{self.config.base_url}{path}"
        headers = self._get_headers()
        last_exc: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._client.request(
                    method,
                    url,
                    headers=headers,
                    **kwargs,
                )
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
                try:
                    error_data = response.json() if response.content else {}
                except (json.JSONDecodeError, ValueError):
                    error_data = {}
                raise PromptGuardError(
                    message=error_data.get("error", {}).get("message", "Request failed"),
                    code=error_data.get("error", {}).get("code", "UNKNOWN"),
                    status_code=response.status_code,
                )

            return response.json()

        if last_exc:
            raise last_exc
        raise PromptGuardError(
            message="Max retries exceeded",
            code="MAX_RETRIES",
            status_code=0,
        )

    def close(self):
        """Close the client"""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class AsyncChatCompletions:
    """Async chat completions API"""

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
            return self._stream_completion(payload)
        return await self._client._request(
            "POST",
            "/chat/completions",
            json=payload,
        )

    async def _stream_completion(
        self,
        payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        payload["stream"] = True
        async with self._client._http.stream(
            "POST",
            f"{self._client.config.base_url}/chat/completions",
            json=payload,
            headers=self._client._get_headers(),
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    yield json.loads(data)


class AsyncChat:
    """Async chat API namespace"""

    def __init__(self, client: "PromptGuardAsync"):
        self.completions = AsyncChatCompletions(client)


class AsyncCompletions:
    """Async legacy completions API"""

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
        return await self._client._request(
            "POST",
            "/completions",
            json=payload,
        )


class AsyncEmbeddings:
    """Async embeddings API"""

    def __init__(self, client: "PromptGuardAsync"):
        self._client = client

    async def create(
        self,
        model: str,
        input: str | list[str],
        **kwargs,
    ) -> dict[str, Any]:
        payload = {"model": model, "input": input, **kwargs}
        return await self._client._request(
            "POST",
            "/embeddings",
            json=payload,
        )


class AsyncSecurity:
    """Async security APIs"""

    def __init__(self, client: "PromptGuardAsync"):
        self._client = client

    async def scan(
        self,
        content: str,
        content_type: str = "prompt",
    ) -> dict[str, Any]:
        return await self._client._request(
            "POST",
            "/security/scan",
            json={"content": content, "type": content_type},
        )

    async def redact(
        self,
        content: str,
        pii_types: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self._client._request(
            "POST",
            "/security/redact",
            json={"content": content, "pii_types": pii_types},
        )


class AsyncScrape:
    """Async scraping API"""

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

    async def batch(
        self,
        urls: list[str],
        **kwargs,
    ) -> dict[str, Any]:
        return await self._client._request(
            "POST",
            "/scrape/batch",
            json={"urls": urls, **kwargs},
        )


class AsyncAgent:
    """Async agent security API"""

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
        return await self._client._request(
            "GET",
            f"/agent/{agent_id}/stats",
        )


class AsyncRedTeam:
    """Async red team testing API"""

    def __init__(self, client: "PromptGuardAsync"):
        self._client = client
        self._base = "/internal/redteam"

    async def list_tests(self) -> dict[str, Any]:
        return await self._client._request(
            "GET",
            f"{self._base}/tests",
        )

    async def run_test(
        self,
        test_name: str,
        target_preset: str = "default",
    ) -> dict[str, Any]:
        return await self._client._request(
            "POST",
            f"{self._base}/test/{test_name}",
            json={"target_preset": target_preset},
        )

    async def run_all(
        self,
        target_preset: str = "default",
    ) -> dict[str, Any]:
        return await self._client._request(
            "POST",
            f"{self._base}/test-all",
            json={"target_preset": target_preset},
        )

    async def run_custom(
        self,
        prompt: str,
        target_preset: str = "default",
    ) -> dict[str, Any]:
        return await self._client._request(
            "POST",
            f"{self._base}/test-custom",
            json={
                "custom_prompt": prompt,
                "target_preset": target_preset,
            },
        )

    async def run_autonomous(
        self,
        budget: int = 100,
        target_preset: str = "default",
        enabled_detectors: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run the autonomous red team agent."""
        payload: dict[str, Any] = {
            "budget": budget,
            "target_preset": target_preset,
        }
        if enabled_detectors is not None:
            payload["enabled_detectors"] = enabled_detectors
        return await self._client._request(
            "POST",
            f"{self._base}/autonomous",
            json=payload,
        )

    async def intelligence_stats(self) -> dict[str, Any]:
        """Get statistics from the Attack Intelligence DB."""
        return await self._client._request(
            "GET",
            f"{self._base}/intelligence/stats",
        )


class PromptGuardAsync:
    """
    Async PromptGuard client -- full parity with the sync client.

    Usage::

        async with PromptGuardAsync(api_key="pg_xxx") as pg:
            resp = await pg.chat.completions.create(
                model="gpt-4",
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
        self.config = config or Config(
            api_key=api_key
            or os.environ.get(
                "PROMPTGUARD_API_KEY",
                "",
            ),
            base_url=base_url
            or os.environ.get(
                "PROMPTGUARD_BASE_URL",
                "https://api.promptguard.co/api/v1/proxy",
            ),
        )

        if not self.config.api_key:
            raise ValueError(
                "API key required. Pass api_key or set PROMPTGUARD_API_KEY environment variable."
            )

        self._http = httpx.AsyncClient(timeout=timeout)

        # OpenAI-compatible APIs
        self.chat = AsyncChat(self)
        self.completions = AsyncCompletions(self)
        self.embeddings = AsyncEmbeddings(self)

        # PromptGuard-specific APIs
        self.security = AsyncSecurity(self)
        self.scrape = AsyncScrape(self)
        self.agent = AsyncAgent(self)
        self.redteam = AsyncRedTeam(self)

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "X-PromptGuard-SDK": _SDK_LANG,
            "X-PromptGuard-Version": _SDK_VERSION,
        }

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Make async API request with retry/backoff."""
        url = f"{self.config.base_url}{path}"
        headers = self._get_headers()
        last_exc: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self._http.request(
                    method,
                    url,
                    headers=headers,
                    **kwargs,
                )
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
                try:
                    error_data = response.json() if response.content else {}
                except (json.JSONDecodeError, ValueError):
                    error_data = {}
                raise PromptGuardError(
                    message=error_data.get("error", {}).get("message", "Request failed"),
                    code=error_data.get("error", {}).get("code", "UNKNOWN"),
                    status_code=response.status_code,
                )

            return response.json()

        if last_exc:
            raise last_exc
        raise PromptGuardError(
            message="Max retries exceeded",
            code="MAX_RETRIES",
            status_code=0,
        )

    async def close(self):
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


class PromptGuardError(Exception):
    """Error from PromptGuard API"""

    def __init__(self, message: str, code: str, status_code: int):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(f"{code}: {message}")
