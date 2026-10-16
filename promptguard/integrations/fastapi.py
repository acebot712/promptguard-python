"""
ASGI middleware - scans prompt-shaped request bodies at the app's edge.

Works with FastAPI, Starlette, or any ASGI app. It imports neither framework.

Usage::

    from fastapi import FastAPI
    from promptguard.integrations.fastapi import PromptGuardMiddleware

    app = FastAPI()
    app.add_middleware(
        PromptGuardMiddleware,
        api_key="pg_live_xxx",          # or PROMPTGUARD_API_KEY
        paths=["/v1/chat"],             # omit to scan every path
        exclude_paths=["/v1/chat/feedback"],
    )

For each ``POST`` / ``PUT`` / ``PATCH`` request on a selected path, the
middleware buffers the body, pulls out its prompt-shaped fields (see
:data:`DEFAULT_PROMPT_FIELDS`), scans them with one ``direction="input"`` Guard
API call, and either answers ``403`` before the app runs or replays the
original bytes to the app unchanged.

This is the choice for an app that **accepts** prompts but calls the LLM
somewhere auto-instrumentation does not patch. With ``promptguard.init()`` also
on, the same text is scanned twice — once here, once at the provider call.

The scan is the async Guard client's ``scan_async``: nothing here blocks the
event loop while the Guard API answers.
"""

import json
import logging
import weakref
from collections.abc import Awaitable, Callable, Iterable, MutableMapping, Sequence
from typing import Any

from promptguard._resolve import resolve_credentials, validate_mode
from promptguard.guard import GuardClient, GuardDecision

logger = logging.getLogger("promptguard")

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

#: The name this middleware reports under ``instrumentation_report()["middleware"]``.
NAME = "fastapi"

#: Top-level JSON body fields scanned by default.
#:
#: Each field is read by the shape of its value, so the one list covers the
#: common LLM request bodies:
#:
#: * a list of message objects (``{"role": ..., "content": ...}``) is scanned
#:   message by message, keeping each role — ``messages`` (OpenAI / Anthropic
#:   chat) and ``input`` (OpenAI Responses API) arrive this way;
#: * a ``content`` that is a list of parts contributes the ``text`` of each
#:   text part; images and other parts are not scanned;
#: * a string, or a list of strings, is one message — ``prompt`` (completions),
#:   ``input``, ``query``, ``question``, ``text``, ``message``, ``content``.
#:
#: ``system`` (Anthropic) and ``instructions`` (Responses API) are scanned with
#: ``role="system"``; everything else that is not a message object is ``user``.
#: Only top-level fields are read. Pass ``prompt_fields=`` to replace the list.
DEFAULT_PROMPT_FIELDS: tuple[str, ...] = (
    "messages",
    "system",
    "instructions",
    "prompt",
    "input",
    "query",
    "question",
    "text",
    "message",
    "content",
)

_SYSTEM_FIELDS = frozenset({"system", "instructions"})

#: Methods whose bodies are scanned by default.
DEFAULT_METHODS: tuple[str, ...] = ("POST", "PUT", "PATCH")

#: The largest body buffered for a scan (10 MiB). A bigger one is refused with
#: ``413`` rather than passed on unscanned: forwarding it would make "pad the
#: body" a way past the guard.
DEFAULT_MAX_BODY_BYTES = 10 * 1024 * 1024

# Every middleware currently alive in this process, for instrumentation_report().
_live: "weakref.WeakSet[PromptGuardMiddleware]" = weakref.WeakSet()


def active_middleware() -> list[str]:
    """Names of the PromptGuard middleware live in this process, sorted."""
    return sorted({mw.name for mw in list(_live)})


def _normalise_paths(paths: Iterable[str] | None) -> tuple[str, ...] | None:
    if paths is None:
        return None
    if isinstance(paths, str):
        # A bare string would otherwise be iterated character by character.
        paths = [paths]
    out = []
    for p in paths:
        p = "/" + p.strip("/") if p.strip("/") else "/"
        out.append(p)
    return tuple(out)


def _path_matches(path: str, prefixes: tuple[str, ...]) -> bool:
    """``True`` when ``path`` is one of ``prefixes`` or below one of them.

    Matching is on whole path segments: ``/v1/chat`` matches ``/v1/chat``,
    ``/v1/chat/`` and ``/v1/chat/completions``, never ``/v1/chatter``.
    ``/`` matches everything.
    """
    for prefix in prefixes:
        if prefix == "/" or path == prefix or path.startswith(prefix + "/"):
            return True
    return False


def _text_of(content: Any) -> str:
    """The scannable text of a ``content``: a string, a text part, or a list of them."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, list):
        return "\n".join(t for t in (_text_of(part) for part in content) if t)
    return ""


def extract_messages(body: Any, fields: Sequence[str]) -> list[dict[str, str]]:
    """The Guard API messages for a parsed JSON ``body``; ``[]`` when none.

    Each list item is read on its own, so one odd item cannot hide the rest: a
    message object becomes a message with its own role, and any other text
    (strings, text parts) joins one message under the field's default role.
    """
    if not isinstance(body, dict):
        return []
    messages: list[dict[str, str]] = []
    for field in fields:
        if field not in body:
            continue
        value = body[field]
        default_role = "system" if field in _SYSTEM_FIELDS else "user"
        loose: list[str] = []
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, dict) and "content" in item:
                text = _text_of(item["content"])
                if text:
                    role = item.get("role")
                    messages.append(
                        {"role": role if isinstance(role, str) else "user", "content": text}
                    )
            else:
                text = _text_of(item)
                if text:
                    loose.append(text)
        if loose:
            messages.append({"role": default_role, "content": "\n".join(loose)})
    return messages


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            decoded: str = value.decode("latin-1")
            return decoded
    return None


class PromptGuardMiddleware:
    """Raw ASGI middleware that scans prompt-shaped request bodies.

    Parameters
    ----------
    app:
        The ASGI app to wrap. ``app.add_middleware(PromptGuardMiddleware, ...)``
        passes it for you.
    api_key, base_url:
        Fall back to ``PROMPTGUARD_API_KEY`` / ``PROMPTGUARD_BASE_URL``.
    mode:
        ``"enforce"`` (default) answers a block with ``403``; ``"monitor"`` logs
        it and lets the request through.
    fail_open:
        ``True`` (default, ADR-0002): when the Guard API cannot be reached or
        returns something unusable, the request proceeds unscanned and a
        ``WARNING`` is logged with ``extra={"promptguard_event":
        "guard_unavailable_fail_open"}``. ``False`` answers ``503`` instead.
    timeout:
        Guard API timeout in seconds (default ``10.0``).
    paths:
        Path prefixes to scan; ``None`` (default) scans every path. Matching is
        on whole segments against the ASGI ``scope["path"]``: ``"/v1/chat"``
        covers ``/v1/chat`` and ``/v1/chat/completions`` but not
        ``/v1/chatter``.
    exclude_paths:
        Path prefixes never scanned, matched the same way. An exclusion wins
        over ``paths``. An excluded request's body is not even buffered.
    methods:
        HTTP methods whose bodies are scanned (default ``POST``, ``PUT``,
        ``PATCH``). Other methods pass straight through.
    prompt_fields:
        Top-level JSON fields to scan; see :data:`DEFAULT_PROMPT_FIELDS`.
    max_body_bytes:
        The largest body buffered for a scan (default 10 MiB). Over it — by
        ``Content-Length`` or while streaming — the request is answered ``413``
        and neither scanned nor passed to the app.

    Bodies that are not JSON are passed on unscanned, except ``text/plain``,
    which is scanned whole as one user message. A redact decision cannot be
    applied to an arbitrary body, so enforce mode refuses it with ``403``
    (ADR-0003) and monitor mode logs it.
    """

    name = NAME

    def __init__(
        self,
        app: ASGIApp,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        mode: str = "enforce",
        fail_open: bool = True,
        timeout: float = 10.0,
        paths: Iterable[str] | None = None,
        exclude_paths: Iterable[str] | None = None,
        methods: Iterable[str] = DEFAULT_METHODS,
        prompt_fields: Sequence[str] = DEFAULT_PROMPT_FIELDS,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    ) -> None:
        resolved_key, resolved_url = resolve_credentials(api_key, base_url)
        validate_mode(mode)
        if max_body_bytes <= 0:
            raise ValueError(f"max_body_bytes must be positive, got {max_body_bytes!r}")
        self.app = app
        self._guard = GuardClient(api_key=resolved_key, base_url=resolved_url, timeout=timeout)
        self._mode = mode
        self._fail_open = fail_open
        self._paths = _normalise_paths(paths)
        self._exclude_paths = _normalise_paths(exclude_paths) or ()
        self._methods = frozenset(m.upper() for m in methods)
        self._prompt_fields = tuple(prompt_fields)
        self._max_body_bytes = max_body_bytes
        _live.add(self)

    def _selected(self, scope: Scope) -> bool:
        if scope.get("method", "").upper() not in self._methods:
            return False
        path = scope.get("path", "")
        if _path_matches(path, self._exclude_paths):
            return False
        return self._paths is None or _path_matches(path, self._paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._selected(scope):
            await self.app(scope, receive, send)
            return

        declared = _header(scope, b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self._max_body_bytes:
            await self._too_large(send)
            return

        chunks: list[bytes] = []
        size = 0
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.disconnect":
                # The client went away before the body finished; there is
                # nobody to answer and nothing complete to scan.
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self._max_body_bytes:
                await self._too_large(send)
                return
            chunks.append(chunk)
            more_body = message.get("more_body", False)
        body = b"".join(chunks)

        messages, model = self._extract(scope, body)
        if messages and not await self._admit(scope, messages, model, send):
            return

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    def _extract(self, scope: Scope, body: bytes) -> tuple[list[dict[str, str]], str | None]:
        """The messages to scan and the body's ``model``, if it names one."""
        if not body:
            return [], None
        content_type = (_header(scope, b"content-type") or "").split(";")[0].strip().lower()
        try:
            parsed = json.loads(body)
        except ValueError:
            if content_type == "text/plain":
                text = body.decode("utf-8", errors="replace").strip()
                return ([{"role": "user", "content": text}] if text else []), None
            return [], None
        model = parsed.get("model") if isinstance(parsed, dict) else None
        return extract_messages(parsed, self._prompt_fields), (
            model if isinstance(model, str) else None
        )

    async def _admit(
        self, scope: Scope, messages: list[dict[str, str]], model: str | None, send: Send
    ) -> bool:
        """Scan ``messages``; answer the request and return ``False`` to refuse it."""
        path = scope.get("path", "")
        try:
            decision = await self._guard.scan_async(
                messages=messages,
                direction="input",
                model=model,
                context={"framework": NAME, "metadata": {"path": path}},
            )
        except Exception as exc:
            if not self._fail_open:
                logger.error(
                    "PromptGuard: Guard API unavailable, refusing %s (fail_open=False): %s",
                    path,
                    exc,
                    extra={"promptguard_event": "guard_unavailable_fail_closed", "path": path},
                )
                await _send_error(
                    send,
                    503,
                    "PromptGuard could not scan this request and is configured to fail closed",
                    "service_unavailable",
                    "guard_unavailable",
                )
                return False
            logger.warning(
                "PromptGuard: Guard API unavailable, allowing %s unscanned (fail_open=True): %s",
                path,
                exc,
                extra={"promptguard_event": "guard_unavailable_fail_open", "path": path},
            )
            return True

        if decision.allowed:
            return True
        if self._mode == "monitor":
            logger.warning(
                "[monitor] PromptGuard would %s %s: %s (event=%s)",
                "block" if decision.blocked else "redact",
                path,
                decision.threat_type,
                decision.event_id,
            )
            return True
        await self._refuse(send, decision)
        return False

    async def _refuse(self, send: Send, decision: GuardDecision) -> None:
        if decision.blocked:
            message, code = "Request blocked by security policy", "request_blocked"
        else:
            message = (
                "Request requires redaction, which PromptGuard cannot apply to this request body"
            )
            code = "redaction_required"
        await _send_error(
            send,
            403,
            message,
            "policy_violation",
            code,
            event_id=decision.event_id,
            details={"threat_type": decision.threat_type, "confidence": decision.confidence},
        )

    async def _too_large(self, send: Send) -> None:
        await _send_error(
            send,
            413,
            f"Request body exceeds the {self._max_body_bytes}-byte limit PromptGuard scans",
            "invalid_request_error",
            "request_too_large",
        )


async def _send_error(
    send: Send, status: int, message: str, error_type: str, code: str, **extra: Any
) -> None:
    """Answer with the platform's documented ``{"error": {...}}`` envelope."""
    payload = json.dumps(
        {"error": {"message": message, "type": error_type, "code": code, **extra}}
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload, "more_body": False})
