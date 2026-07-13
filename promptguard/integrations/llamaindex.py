"""
LlamaIndex integration - PromptGuardCallbackHandler for LlamaIndex.

Hooks into LlamaIndex's callback system to scan LLM prompts and responses
with rich pipeline context.

Usage::

    from promptguard.integrations.llamaindex import PromptGuardCallbackHandler
    from llama_index.core.callbacks import CallbackManager

    pg_handler = PromptGuardCallbackHandler(api_key="pg_live_xxx")
    callback_manager = CallbackManager([pg_handler])

    from llama_index.core import Settings
    Settings.callback_manager = callback_manager
"""

import logging
from typing import Any

from promptguard._resolve import resolve_credentials, validate_mode
from promptguard.guard import GuardClient, GuardDecision, PromptGuardBlockedError

logger = logging.getLogger("promptguard")

_LLM_EVENT = "llm"
_QUERY_EVENT = "query"


class PromptGuardCallbackHandler:
    """LlamaIndex callback handler that scans LLM I/O via the Guard API.

    Implements the LlamaIndex ``BaseCallbackHandler`` interface without
    importing LlamaIndex at module level.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        mode: str = "enforce",
        scan_responses: bool = True,
        fail_open: bool = True,
        timeout: float = 10.0,
        event_starts_to_ignore: list[str] | None = None,
        event_ends_to_ignore: list[str] | None = None,
    ):
        """Construct the handler.

        Parameters
        ----------
        mode:
            ``"enforce"`` to block policy violations, ``"monitor"`` to log only.
        scan_responses:
            Defaults to ``True`` here — unlike ``promptguard.init()`` which
            defaults to ``False``. The callback already receives ``on_event_end``
            events for free, so scanning responses adds no extra call surface and
            richer coverage is the expected behaviour when opting into a
            framework integration. ``init()`` stays conservative (off) because it
            is the zero-config drop-in and response scanning doubles the Guard
            API round-trips per LLM call.
        timeout:
            HTTP timeout (seconds) for Guard API calls (default 10s, matching the
            Guard scan path; the proxy client uses 30s because it fronts the full
            upstream LLM call).
        """
        resolved_key, resolved_url = resolve_credentials(api_key, base_url)
        validate_mode(mode)
        self._guard = GuardClient(
            api_key=resolved_key,
            base_url=resolved_url,
            timeout=timeout,
        )
        self._mode = mode
        self._scan_responses = scan_responses
        self._fail_open = fail_open
        self.event_starts_to_ignore = event_starts_to_ignore or []
        self.event_ends_to_ignore = event_ends_to_ignore or []
        self._event_context: dict[str, dict[str, Any]] = {}

    def on_event_start(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        if event_type in self.event_starts_to_ignore:
            return event_id

        self._event_context[event_id] = {
            "event_type": event_type,
            "parent_id": parent_id,
        }

        if event_type == _LLM_EVENT and payload:
            self._scan_llm_start(payload, event_id)
        elif event_type == _QUERY_EVENT and payload:
            self._scan_query_start(payload, event_id)

        return event_id

    def on_event_end(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if event_type in self.event_ends_to_ignore:
            return

        if self._scan_responses:
            if event_type == _LLM_EVENT and payload:
                self._scan_llm_end(payload, event_id)
            elif event_type == _QUERY_EVENT and payload:
                self._scan_query_end(payload, event_id)

        self._event_context.pop(event_id, None)

    def start_trace(self, trace_id: str | None = None) -> None:
        pass

    def end_trace(
        self,
        trace_id: str | None = None,
        trace_map: dict[str, list[str]] | None = None,
    ) -> None:
        pass

    # -- LLM event scanning --------------------------------------------------

    def _scan_llm_start(self, payload: dict[str, Any], event_id: str) -> None:
        messages = self._extract_messages_from_payload(payload)
        if not messages:
            return

        model = payload.get("model_name") or payload.get("model") or "unknown"
        context = {
            "framework": "llamaindex",
            "session_id": event_id,
            "metadata": {"event_type": _LLM_EVENT, "stage": "start"},
        }

        decision = self._safe_scan(messages, "input", model, context)
        self._handle_decision(decision, event_id)

    def _scan_llm_end(self, payload: dict[str, Any], event_id: str) -> None:
        text = self._extract_response_from_payload(payload)
        if not text:
            return

        context = {
            "framework": "llamaindex",
            "session_id": event_id,
            "metadata": {"event_type": _LLM_EVENT, "stage": "end"},
        }

        messages = [{"role": "assistant", "content": text}]
        decision = self._safe_scan(messages, "output", None, context)
        self._handle_decision(decision, event_id)

    def _scan_query_start(self, payload: dict[str, Any], event_id: str) -> None:
        query_str = payload.get("query_str") or payload.get("query")
        if not query_str or not isinstance(query_str, str):
            return

        messages = [{"role": "user", "content": query_str}]
        context = {
            "framework": "llamaindex",
            "session_id": event_id,
            "metadata": {"event_type": _QUERY_EVENT, "stage": "start"},
        }

        decision = self._safe_scan(messages, "input", None, context)
        self._handle_decision(decision, event_id)

    def _scan_query_end(self, payload: dict[str, Any], event_id: str) -> None:
        response = payload.get("response")
        if not response:
            return

        text = str(response) if not isinstance(response, str) else response
        if not text:
            return

        messages = [{"role": "assistant", "content": text}]
        context = {
            "framework": "llamaindex",
            "session_id": event_id,
            "metadata": {"event_type": _QUERY_EVENT, "stage": "end"},
        }

        decision = self._safe_scan(messages, "output", None, context)
        self._handle_decision(decision, event_id)

    # -- Helpers -------------------------------------------------------------

    def _extract_messages_from_payload(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        messages = []

        raw_messages = payload.get("messages")
        if raw_messages:
            for msg in raw_messages:
                if isinstance(msg, dict):
                    messages.append(
                        {
                            "role": msg.get("role", "user"),
                            "content": str(msg.get("content", "")),
                        }
                    )
                elif hasattr(msg, "role") and hasattr(msg, "content"):
                    role = str(msg.role)
                    role_map = {
                        "MessageRole.USER": "user",
                        "MessageRole.SYSTEM": "system",
                        "MessageRole.ASSISTANT": "assistant",
                    }
                    messages.append(
                        {
                            "role": role_map.get(
                                role,
                                role.split(".")[-1].lower() if "." in role else role,
                            ),
                            "content": str(msg.content),
                        }
                    )
            return messages

        prompt = payload.get("prompt") or payload.get("formatted_prompt")
        if prompt and isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})

        template = payload.get("template")
        if template and isinstance(template, str) and not messages:
            messages.append({"role": "user", "content": template})

        return messages

    def _extract_response_from_payload(self, payload: dict[str, Any]) -> str | None:
        response = payload.get("response") or payload.get("completion")
        if response:
            if isinstance(response, str):
                return response
            if hasattr(response, "text"):
                resp_text: str | None = response.text
                return resp_text
            if hasattr(response, "message") and hasattr(response.message, "content"):
                return str(response.message.content)
            return str(response)

        raw = payload.get("raw")
        if raw and hasattr(raw, "text"):
            raw_text: str | None = raw.text
            return raw_text

        return None

    def _safe_scan(
        self,
        messages: list[dict[str, str]],
        direction: str,
        model: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> GuardDecision | None:
        try:
            return self._guard.scan(
                messages=messages,
                direction=direction,
                model=model,
                context=context,
            )
        except Exception:
            if not self._fail_open:
                raise
            logger.warning("Guard API unavailable (fail_open=True)")
            return None

    def _handle_decision(self, decision: GuardDecision | None, event_id: str) -> None:
        if decision is None:
            return

        if decision.blocked:
            if self._mode == "enforce":
                raise PromptGuardBlockedError(decision)
            logger.warning(
                "[monitor] PromptGuard would block: %s (event=%s, llamaindex_event=%s)",
                decision.threat_type,
                decision.event_id,
                event_id,
            )

        if decision.redacted:
            # The callback observes events but cannot rewrite prompts/responses
            # in flight, so a redact decision can't be applied. In enforce mode
            # we block rather than forward unredacted content; monitor warns.
            if self._mode == "enforce":
                logger.error(
                    "PromptGuard: redaction required but the LlamaIndex callback "
                    "cannot rewrite in-flight content; blocking (threat=%s, "
                    "event=%s, llamaindex_event=%s)",
                    decision.threat_type,
                    decision.event_id,
                    event_id,
                )
                raise PromptGuardBlockedError(decision)
            logger.warning(
                "[monitor] PromptGuard would redact content (event=%s)", decision.event_id
            )


def _require_fn_component() -> Any:
    """Lazily import ``FnComponent`` from ``llama_index.core``.

    Kept out of module scope so importing
    ``promptguard.integrations.llamaindex`` never requires LlamaIndex to be
    installed (the callback handler duck-types the interface). Only the inline
    query-pipeline component needs the real class.
    """
    try:
        from llama_index.core.query_pipeline import FnComponent
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "The LlamaIndex query guard requires 'llama-index-core', which is "
            "not installed. Install it with: pip install promptguard-sdk[llamaindex]"
        ) from exc
    return FnComponent


class PromptGuardQueryGuard:
    """Inline LlamaIndex guard that scans (and can redact) a query string.

    Unlike :class:`PromptGuardCallbackHandler`, which only *observes* events and
    can therefore block but never rewrite them, this guard is a pipeline
    component that sits **in** the data flow. It scans the query before
    retrieval/synthesis, raising :class:`PromptGuardBlockedError` on a ``block``
    decision and returning the redacted query on a ``redact`` decision. Drop it
    at the head of a ``QueryPipeline``::

        from promptguard.integrations.llamaindex import PromptGuardQueryGuard
        from llama_index.core.query_pipeline import QueryPipeline

        guard = PromptGuardQueryGuard(api_key="pg_live_xxx")
        pipeline = QueryPipeline(chain=[guard.as_query_component(), retriever])
        pipeline.run(input="user question")

    ``guard_query`` is also usable directly as a plain preprocessor function
    (no LlamaIndex import required) if you are not building a ``QueryPipeline``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        mode: str = "enforce",
        fail_open: bool = True,
        timeout: float = 10.0,
    ):
        resolved_key, resolved_url = resolve_credentials(api_key, base_url)
        validate_mode(mode)
        self._guard = GuardClient(
            api_key=resolved_key,
            base_url=resolved_url,
            timeout=timeout,
        )
        self._mode = mode
        self._fail_open = fail_open

    def guard_query(self, query_str: str) -> str:
        """Scan ``query_str``; return it (possibly redacted) or raise on a block."""
        text = str(query_str)
        messages = [{"role": "user", "content": text}]
        context = {
            "framework": "llamaindex",
            "metadata": {"event_type": _QUERY_EVENT, "stage": "preprocess"},
        }
        decision = self._safe_scan(messages, "input", None, context)
        if decision is None or decision.allowed:
            return text

        if decision.blocked:
            if self._mode == "enforce":
                raise PromptGuardBlockedError(decision)
            logger.warning(
                "[monitor] PromptGuard would block query: %s (event=%s)",
                decision.threat_type,
                decision.event_id,
            )
            return text

        # redact
        if self._mode != "enforce":
            logger.warning(
                "[monitor] PromptGuard would redact query: %s (event=%s)",
                decision.threat_type,
                decision.event_id,
            )
            return text

        redacted = decision.redacted_messages or []
        if redacted and isinstance(redacted[0].get("content"), str):
            return redacted[0]["content"]

        # No usable redaction payload — fail safe rather than forward the
        # flagged query unredacted.
        logger.error(
            "PromptGuard: redaction required but no redacted query was returned; "
            "blocking (threat=%s, event=%s)",
            decision.threat_type,
            decision.event_id,
        )
        raise PromptGuardBlockedError(decision)

    def as_query_component(self) -> Any:
        """Wrap :meth:`guard_query` in a LlamaIndex ``FnComponent``."""
        fn_component = _require_fn_component()
        return fn_component(fn=self.guard_query)

    def _safe_scan(
        self,
        messages: list[dict[str, str]],
        direction: str,
        model: str | None,
        context: dict[str, Any] | None,
    ) -> GuardDecision | None:
        try:
            return self._guard.scan(
                messages=messages,
                direction=direction,
                model=model,
                context=context,
            )
        except Exception:
            if not self._fail_open:
                raise
            logger.warning("Guard API unavailable (fail_open=True)")
            return None


# Framework-disambiguating alias so callers can import both the LangChain and
# LlamaIndex handlers without an alias clash (both historically export
# ``PromptGuardCallbackHandler``). The original name is kept for back-compat.
LlamaIndexCallbackHandler = PromptGuardCallbackHandler

__all__ = [
    "LlamaIndexCallbackHandler",
    "PromptGuardCallbackHandler",
    "PromptGuardQueryGuard",
]
