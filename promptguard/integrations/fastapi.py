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

For each request with a body on a selected path, the middleware buffers the
body, pulls out its prompt-shaped fields (see :data:`DEFAULT_PROMPT_FIELDS`),
scans them with one ``direction="input"`` Guard API call, and either answers
before the app runs or replays the original bytes to the app unchanged.

This is the choice for an app that **accepts** prompts but calls the LLM
somewhere auto-instrumentation does not patch. With ``promptguard.init()`` also
on, the same text is scanned twice — once here, once at the provider call.

The scan is the async Guard client's ``scan_async``, and a body larger than
:data:`OFF_LOOP_BODY_BYTES` is decoded and parsed in a worker thread, so
nothing here holds the event loop while a request is inspected.
"""

import codecs
import json
import logging
import zlib
from collections.abc import Awaitable, Callable, Iterable, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast
from urllib.parse import parse_qsl

import anyio.to_thread

from promptguard._resolve import resolve_credentials, validate_mode
from promptguard._session import as_session_id, with_session_id
from promptguard.auto import _register_middleware
from promptguard.guard import GuardApiError, GuardClient, GuardDecision

logger = logging.getLogger("promptguard")

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

#: The name this middleware reports under ``instrumentation_report()["middleware"]``.
NAME = "fastapi"

#: Top-level body fields scanned by default.
#:
#: Each field is read by the shape of its value, so the one list covers the
#: common LLM request bodies:
#:
#: * a list of message objects (anything with ``role``, ``content`` or
#:   ``tool_calls``) is scanned message by message, keeping each role —
#:   ``messages`` (OpenAI / Anthropic chat) and ``input`` (OpenAI Responses
#:   API) arrive this way; a message's ``tool_calls`` go to the Guard API as
#:   tool calls;
#: * ``content`` that is a list of blocks is passed to the Guard API as those
#:   blocks, so images, documents and tool-use blocks keep their coverage;
#: * a string is one message, and a list of strings is one message of text
#:   blocks.
#:
#: The fields in :data:`SYSTEM_PROMPT_FIELDS` are scanned with
#: ``role="system"``; anything else that is not a message object is ``user``.
#: Only top-level fields are read. A top-level JSON array is read item by
#: item: a message object is a message, an object has these fields read, a
#: string is user text. Pass ``prompt_fields=`` to replace the list.
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

#: The prompt fields whose loose text is a system prompt (Anthropic's
#: ``system``, the Responses API's ``instructions``).
SYSTEM_PROMPT_FIELDS: frozenset[str] = frozenset({"system", "instructions"})

#: The largest body buffered for a scan, raw and after decompression (10 MiB).
#: A bigger one is refused with ``413`` rather than passed on unscanned:
#: forwarding it would make "pad the body" a way past the guard.
DEFAULT_MAX_BODY_BYTES = 10 * 1024 * 1024

#: Bodies larger than this (64 KiB) are decompressed, decoded and parsed in a
#: worker thread rather than on the event loop.
OFF_LOOP_BODY_BYTES = 64 * 1024

# Guard API statuses that mean the request itself is the problem (too large,
# malformed). Failing open on these would let a caller pad a prompt past the
# Guard's limits to get it through, so they are refused in every mode.
_REQUEST_FAULT_STATUSES = frozenset({400, 413, 422})
# Statuses that mean this deployment is misconfigured (a bad key). They follow
# fail_open like an outage, but are logged at ERROR once: verify() is the tool
# that catches them before production does.
_AUTH_FAULT_STATUSES = frozenset({401, 403})
# Markers of a size complaint in a Guard 422 body. The guard router rejects an
# over-long message with "content exceeds N characters" and an over-long list
# with pydantic's ``too_long``.
_SIZE_FAULT_MARKERS = ("exceeds", "too_long", "too large")


@dataclass
class _Scan:
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model: str | None = None


@dataclass
class _Refusal:
    status: int
    code: str
    message: str
    error_type: str = "invalid_request_error"
    # Refused only in enforce mode; monitor logs it and passes the request on.
    enforce_only: bool = True


class _TooLarge(Exception):
    pass


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


def _route_path(scope: Scope) -> str:
    """The path the app's own routes see: ``scope["path"]`` less ``root_path``.

    Mirrors Starlette's ``get_route_path``. Behind a proxy prefix or a
    ``Mount``, ``scope["path"]`` carries the prefix and ``root_path`` names it;
    ``paths`` are written against the app's routes, so the prefix comes off.
    """
    path: str = scope.get("path", "")
    root_path: str = scope.get("root_path", "")
    if not root_path or not path.startswith(root_path):
        return path or "/"
    if path == root_path:
        return "/"
    if path[len(root_path)] == "/":
        return path[len(root_path) :]
    return path


def _path_matches(path: str, prefixes: tuple[str, ...]) -> bool:
    """``True`` when ``path`` is one of ``prefixes`` or below one of them.

    Matching is on whole path segments: ``/v1/chat`` matches ``/v1/chat``,
    ``/v1/chat/`` and ``/v1/chat/completions``, never ``/v1/chatter``.
    ``/`` matches everything.
    """
    return any(
        prefix == "/" or path == prefix or path.startswith(prefix + "/") for prefix in prefixes
    )


# -- Reading the body --------------------------------------------------------


def _block(item: Any) -> dict[str, Any] | None:
    """One content block for the Guard API, or ``None`` for nothing to scan."""
    if isinstance(item, str):
        return {"type": "text", "text": item} if item else None
    if isinstance(item, dict):
        if "type" not in item and isinstance(item.get("text"), str):
            return {"type": "text", "text": item["text"]}
        return item
    return None


def _content(value: Any) -> str | list[dict[str, Any]] | None:
    """A message ``content`` as the Guard API takes it: a string or blocks."""
    if isinstance(value, str):
        return value or None
    items = value if isinstance(value, list) else [value]
    blocks = [b for b in (_block(v) for v in items) if b]
    return blocks or None


def _is_message(item: Any) -> bool:
    return isinstance(item, dict) and any(k in item for k in ("role", "content", "tool_calls"))


def _add_message(item: dict[str, Any], scan: _Scan, default_role: str = "user") -> None:
    calls = item.get("tool_calls")
    if isinstance(calls, list):
        scan.tool_calls.extend(c for c in calls if isinstance(c, dict))
    content = _content(item.get("content"))
    if content is not None:
        role = item.get("role")
        scan.messages.append(
            {"role": role if isinstance(role, str) else default_role, "content": content}
        )


def _add_fields(body: dict[str, Any], fields: Sequence[str], scan: _Scan) -> None:
    """Read the prompt ``fields`` of one object into ``scan``.

    Each list item is read on its own, so one odd item cannot hide the rest: a
    message object becomes a message with its own role, and anything else
    joins one message of blocks under the field's default role.
    """
    for name in fields:
        if name not in body:
            continue
        value = body[name]
        default_role = "system" if name in SYSTEM_PROMPT_FIELDS else "user"
        if isinstance(value, str):
            if value:
                scan.messages.append({"role": default_role, "content": value})
            continue
        loose: list[Any] = []
        for item in value if isinstance(value, list) else [value]:
            if _is_message(item):
                _add_message(item, scan, default_role)
            else:
                loose.append(item)
        content = _content(loose)
        if content is not None:
            scan.messages.append({"role": default_role, "content": content})


def extract_scan(body: Any, fields: Sequence[str]) -> _Scan:
    """What to scan in a parsed body (an object, or a top-level array)."""
    scan = _Scan(messages=[])
    if isinstance(body, dict):
        _add_fields(body, fields, scan)
        model = body.get("model")
        scan.model = model if isinstance(model, str) else None
    elif isinstance(body, list):
        for item in body:
            if _is_message(item):
                _add_message(item, scan)
            elif isinstance(item, dict):
                _add_fields(item, fields, scan)
            elif isinstance(item, str) and item:
                scan.messages.append({"role": "user", "content": item})
    return scan


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            decoded: str = value.decode("latin-1")
            return decoded
    return None


def _header_reader(name: bytes) -> Callable[[Scope], str | None]:
    """A ``session_id`` function that reads one request header."""
    return lambda scope: _header(scope, name)


def _media_type(header: str | None) -> tuple[str, str | None]:
    """``(media type, charset)`` from a ``Content-Type`` header."""
    if not header:
        return "", None
    media, *params = header.split(";")
    charset = None
    for param in params:
        key, _, value = param.partition("=")
        if key.strip().lower() == "charset":
            charset = value.strip().strip('"').strip("'") or None
    return media.strip().lower(), charset


def _inflate(data: bytes, wbits: int, limit: int) -> bytes:
    """Decompress ``data``, refusing to produce more than ``limit`` bytes."""
    out = bytearray()
    while data:
        d = zlib.decompressobj(wbits)
        out += d.decompress(data, limit + 1 - len(out))
        if len(out) > limit or d.unconsumed_tail:
            raise _TooLarge
        out += d.flush()
        if len(out) > limit:
            raise _TooLarge
        if not d.eof:
            raise zlib.error("truncated stream")
        # gzip allows several members back to back.
        data = d.unused_data if wbits == 31 else b""
    return bytes(out)


def _decompress(body: bytes, encoding: str | None, limit: int) -> bytes | _Refusal:
    for coding in reversed([c.strip().lower() for c in (encoding or "").split(",") if c.strip()]):
        try:
            if coding == "identity":
                continue
            if coding in ("gzip", "x-gzip"):
                body = _inflate(body, 31, limit)
            elif coding == "deflate":
                # "deflate" means zlib-wrapped, but many clients send it raw.
                try:
                    body = _inflate(body, 15, limit)
                except zlib.error:
                    body = _inflate(body, -15, limit)
            else:
                return _Refusal(
                    415,
                    "unsupported_content_encoding",
                    f"PromptGuard cannot scan a {coding!r}-encoded body",
                )
        except _TooLarge:
            return _Refusal(
                413,
                "request_too_large",
                f"Decompressed request body exceeds the {limit}-byte limit PromptGuard scans",
                enforce_only=False,
            )
        except zlib.error:
            return _Refusal(400, "invalid_body", f"The {coding}-encoded body is corrupt")
    return body


def _decode(body: bytes, charset: str) -> str | _Refusal:
    try:
        codecs.lookup(charset)
    except LookupError:
        return _Refusal(415, "unsupported_charset", f"Unknown charset {charset!r}")
    try:
        return body.decode(charset)
    except (UnicodeDecodeError, ValueError):
        return _Refusal(400, "invalid_body", f"The body is not valid {charset}")


def _read_body(
    body: bytes,
    content_type: str | None,
    content_encoding: str | None,
    fields: Sequence[str],
    limit: int,
) -> _Scan | _Refusal | None:
    """Decompress, decode and parse ``body``; ``None`` when nothing is scannable.

    Synchronous and side-effect free, so it can run in a worker thread.
    """
    decompressed = _decompress(body, content_encoding, limit)
    if isinstance(decompressed, _Refusal):
        return decompressed
    if not decompressed:
        return None
    media, charset = _media_type(content_type)

    if media == "text/plain":
        text = _decode(decompressed, charset or "utf-8")
        if isinstance(text, _Refusal):
            return text
        text = text.strip()
        return _Scan(messages=[{"role": "user", "content": text}]) if text else None

    if media == "application/x-www-form-urlencoded":
        text = _decode(decompressed, charset or "utf-8")
        if isinstance(text, _Refusal):
            return text
        form: dict[str, list[str]] = {}
        for key, value in parse_qsl(text, keep_blank_values=True):
            form.setdefault(key, []).append(value)
        flattened: dict[str, Any] = {k: v[0] if len(v) == 1 else v for k, v in form.items()}
        return extract_scan(flattened, fields)

    declared_json = media == "application/json" or media.endswith("+json")
    if not declared_json and media:
        # multipart/form-data, binary uploads and other media are not read.
        return None
    try:
        if charset:
            text = _decode(decompressed, charset)
            if isinstance(text, _Refusal):
                return text
            parsed = json.loads(text)
        else:
            parsed = json.loads(decompressed)
    except RecursionError:
        return _Refusal(400, "invalid_json", "The JSON body is nested too deeply to scan")
    except ValueError:
        if not declared_json:
            # No Content-Type and not JSON: nothing this middleware can read.
            return None
        return _Refusal(400, "invalid_json", "The body is not valid JSON")
    return extract_scan(parsed, fields)


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
        Governs a Guard API that cannot answer: a transport error, a timeout, a
        5xx or a 429 (and a 401/403, which is also logged at ``ERROR`` once).
        ``True`` (default, ADR-0002) lets the request through unscanned and logs
        a ``WARNING`` with ``extra={"promptguard_event":
        "guard_unavailable_fail_open"}``; ``False`` answers ``503``. A Guard
        ``400``/``413``/``422`` is caused by the request itself and is refused
        in every mode: ``413`` for a size error, ``400`` otherwise.
    timeout:
        Guard API timeout in seconds (default ``10.0``).
    paths:
        Path prefixes to scan; ``None`` (default) scans every path, and an empty
        list is refused. Matching is on whole segments of the app's route path
        (``scope["path"]`` less ``root_path``, as Starlette routes it):
        ``"/v1/chat"`` covers ``/v1/chat`` and ``/v1/chat/completions`` but not
        ``/v1/chatter``.
    exclude_paths:
        Path prefixes never scanned, matched the same way. An exclusion wins
        over ``paths``. An excluded request's body is not even buffered.
    methods:
        HTTP methods to scan; ``None`` (default) scans any method that carries
        a body, ``GET`` and ``DELETE`` included.
    prompt_fields:
        Top-level fields to scan; see :data:`DEFAULT_PROMPT_FIELDS`.
    max_body_bytes:
        The largest body buffered for a scan, raw or decompressed; see
        :data:`DEFAULT_MAX_BODY_BYTES`. Over it the request is answered
        ``413`` and neither scanned nor passed to the app.
    session_id:
        A function of the ASGI ``scope`` returning the Conversation the request
        is a turn of, or ``None``. Read it from the authenticated session your
        auth middleware put in the scope (``scope["user"]``,
        ``scope["session"]``, ``scope["state"]``), not from anything the
        client controls. A function that raises is logged and the request is
        scanned without a session id.
    session_id_header:
        Shorthand for a ``session_id`` that reads one request header, e.g.
        ``"X-Conversation-Id"``. Name only a header your own auth layer or a
        trusted proxy sets and strips from client requests: a value a browser
        can choose lets a client pick which Conversation its turns land in.
        Not both at once.

    A request is not a conversation by itself, so with neither (the default),
    or when they name none, no session id is sent. An id over 255 characters
    is dropped.

    Bodies are read by ``Content-Type``: JSON (in its declared charset),
    ``text/plain`` (scanned whole, as one user message) and
    ``application/x-www-form-urlencoded`` (its fields). ``gzip`` and
    ``deflate`` bodies are decompressed first. ``multipart/form-data`` and
    other media are passed on unscanned. In enforce mode a body that cannot
    be read — an unknown ``Content-Encoding`` (``415``) or charset (``415``),
    undecodable text or invalid JSON (``400``) — is refused; monitor mode logs
    it and passes it on.

    A redact decision cannot be applied to an arbitrary body, so enforce mode
    refuses it with ``403`` and monitor mode logs it. That is the rule ADR-0003
    sets for a patched call auto-instrumentation cannot rewrite, applied here
    by analogy.
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
        methods: Iterable[str] | None = None,
        prompt_fields: Sequence[str] = DEFAULT_PROMPT_FIELDS,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        session_id: Callable[[Scope], str | None] | None = None,
        session_id_header: str | None = None,
    ) -> None:
        resolved_key, resolved_url = resolve_credentials(api_key, base_url)
        validate_mode(mode)
        if max_body_bytes <= 0:
            raise ValueError(f"max_body_bytes must be positive, got {max_body_bytes!r}")
        normalised_paths = _normalise_paths(paths)
        if normalised_paths == ():
            raise ValueError(
                "paths=[] would scan nothing; pass None to scan every path, "
                "or list the paths to scan"
            )
        self.app = app
        self._guard = GuardClient(api_key=resolved_key, base_url=resolved_url, timeout=timeout)
        self._mode = mode
        self._fail_open = fail_open
        self._paths = normalised_paths
        self._exclude_paths = _normalise_paths(exclude_paths) or ()
        self._methods = None if methods is None else frozenset(m.upper() for m in methods)
        self._prompt_fields = tuple(prompt_fields)
        self._max_body_bytes = max_body_bytes
        if session_id_header is not None and not session_id_header.strip():
            raise ValueError("session_id_header must name a header, or be None")
        if session_id is not None and session_id_header is not None:
            raise ValueError("pass session_id or session_id_header, not both")
        if session_id_header is not None:
            session_id = _header_reader(session_id_header.strip().lower().encode("latin-1"))
        self._session_id = session_id
        self._wrapping = False
        self._auth_failure_logged = False

    def _selected(self, scope: Scope) -> bool:
        if self._methods is not None and scope.get("method", "").upper() not in self._methods:
            return False
        path = _route_path(scope)
        if _path_matches(path, self._exclude_paths):
            return False
        return self._paths is None or _path_matches(path, self._paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self._wrapping:
            # Called means in front of an app: Starlette builds its middleware
            # stack on the first call (lifespan at startup, else the first
            # request), so a constructed-but-unused instance is never reported.
            self._wrapping = True
            _register_middleware(self)

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

        if body and not await self._admit(scope, body, send):
            return

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    @staticmethod
    def _read_session_id(
        read: Callable[[Scope], str | None], scope: Scope, path: str
    ) -> str | None:
        try:
            return as_session_id(read(scope))
        except Exception:
            logger.warning(
                "PromptGuard: session_id raised; scanning %s without a session id",
                path,
                exc_info=True,
                extra={"promptguard_event": "session_id_failed", "path": path},
            )
            return None

    async def _admit(self, scope: Scope, body: bytes, send: Send) -> bool:
        """Read and scan ``body``; answer the request and return ``False`` to refuse it."""
        path = _route_path(scope)
        args = (
            body,
            _header(scope, b"content-type"),
            _header(scope, b"content-encoding"),
            self._prompt_fields,
            self._max_body_bytes,
        )
        if len(body) > OFF_LOOP_BODY_BYTES:
            read = await anyio.to_thread.run_sync(_read_body, *args)
        else:
            read = _read_body(*args)

        if isinstance(read, _Refusal):
            if read.enforce_only and self._mode == "monitor":
                logger.warning(
                    "[monitor] PromptGuard would refuse %s unscanned: %s",
                    path,
                    read.message,
                    extra={"promptguard_event": "body_unscannable", "path": path},
                )
                return True
            await _send_error(send, read.status, read.message, read.error_type, read.code)
            return False
        if read is None or not (read.messages or read.tool_calls):
            return True

        context: dict[str, Any] = {"framework": NAME, "metadata": {"path": path}}
        if read.tool_calls:
            context["tool_calls"] = read.tool_calls
        if self._session_id is not None:
            with_session_id(context, self._read_session_id(self._session_id, scope, path))
        try:
            decision = await self._guard.scan_async(
                # The Guard API takes content blocks as well as strings.
                messages=cast("list[dict[str, str]]", read.messages),
                direction="input",
                model=read.model,
                context=context,
            )
        except Exception as exc:
            return await self._on_guard_error(exc, path, send)

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

    async def _on_guard_error(self, exc: Exception, path: str, send: Send) -> bool:
        status = exc.status_code if isinstance(exc, GuardApiError) else None
        if status in _REQUEST_FAULT_STATUSES:
            size_fault = status == 413 or any(m in str(exc) for m in _SIZE_FAULT_MARKERS)
            logger.warning(
                "PromptGuard: the Guard API rejected %s (%s); refusing it: %s",
                path,
                status,
                exc,
                extra={"promptguard_event": "guard_rejected_request", "path": path},
            )
            if size_fault:
                await _send_error(
                    send,
                    413,
                    "Request content exceeds what PromptGuard can scan",
                    "invalid_request_error",
                    "request_too_large",
                )
            else:
                await _send_error(
                    send,
                    400,
                    "PromptGuard could not scan this request",
                    "invalid_request_error",
                    "request_unscannable",
                )
            return False

        if status in _AUTH_FAULT_STATUSES and not self._auth_failure_logged:
            self._auth_failure_logged = True
            logger.error(
                "PromptGuard: the Guard API refused this middleware's API key (%s). "
                "Requests are %s until it is fixed; run promptguard.verify() to check "
                "the setup.",
                status,
                "passing unscanned" if self._fail_open else "being refused",
                extra={"promptguard_event": "guard_auth_failed", "path": path},
            )

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
