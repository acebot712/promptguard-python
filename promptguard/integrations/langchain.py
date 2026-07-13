"""
LangChain integration - PromptGuardCallbackHandler.

Implements LangChain's ``BaseCallbackHandler`` to scan prompts before
LLM calls and responses after, with rich context about chains, tools,
and agent steps.

Usage::

    from promptguard.integrations.langchain import PromptGuardCallbackHandler

    handler = PromptGuardCallbackHandler(api_key="pg_live_xxx")

    # Attach to a single LLM
    llm = ChatOpenAI(model="gpt-5-nano", callbacks=[handler])

    # Or use globally with any chain / agent
    chain.invoke({"input": "..."}, config={"callbacks": [handler]})
"""

import logging
from typing import Any
from uuid import UUID

from promptguard._resolve import resolve_credentials, validate_mode
from promptguard.guard import GuardClient, GuardDecision, PromptGuardBlockedError

logger = logging.getLogger("promptguard")

# Keys we look inside when a chain hands the guard a ``dict`` input (the common
# ``chain.invoke({"input": "..."})`` shape) so the raw user text can be scanned.
_DICT_TEXT_KEYS = ("input", "query", "question", "text", "content", "prompt")


class PromptGuardCallbackHandler:
    """LangChain callback handler that scans LLM I/O via the Guard API.

    This class implements the LangChain ``BaseCallbackHandler`` interface
    without importing LangChain at module level (so the SDK doesn't have
    a hard dependency on it).
    """

    raise_error = True

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        mode: str = "enforce",
        scan_responses: bool = True,
        fail_open: bool = True,
        timeout: float = 10.0,
    ):
        """Construct the handler.

        Parameters
        ----------
        mode:
            ``"enforce"`` to block policy violations, ``"monitor"`` to log only.
        scan_responses:
            Defaults to ``True`` here — unlike ``promptguard.init()`` which
            defaults to ``False``. The callback already receives ``on_llm_end``
            events for free, so scanning responses adds no extra call surface
            and richer coverage is the expected behaviour when a user opts into
            a framework integration. ``init()`` stays conservative (off) because
            it is the zero-config drop-in and response scanning doubles the
            Guard API round-trips per LLM call.
        timeout:
            HTTP timeout (seconds) for Guard API calls (default 10s, matching
            the Guard scan path; the proxy client uses 30s because it fronts the
            full upstream LLM call).
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
        self._chain_context: dict[str, dict[str, Any]] = {}

    # -- LLM callbacks -------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        messages = [{"role": "user", "content": p} for p in prompts]
        model = serialized.get("kwargs", {}).get("model_name") or serialized.get("id", [""])[-1]
        context = self._build_context(run_id, parent_run_id, "llm", serialized, tags, metadata)

        decision = self._safe_scan(messages, "input", model, context)
        self._handle_decision(decision, run_id)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        guard_messages = []
        for message_list in messages:
            for msg in message_list:
                if isinstance(msg, dict):
                    guard_messages.append(
                        {
                            "role": msg.get("role", "user"),
                            "content": str(msg.get("content", "")),
                        }
                    )
                elif hasattr(msg, "type") and hasattr(msg, "content"):
                    role = getattr(msg, "type", "user")
                    role_map = {"human": "user", "ai": "assistant", "system": "system"}
                    guard_messages.append(
                        {
                            "role": role_map.get(role, role),
                            "content": str(msg.content),
                        }
                    )
                elif hasattr(msg, "role") and hasattr(msg, "content"):
                    guard_messages.append(
                        {
                            "role": str(msg.role),
                            "content": str(msg.content),
                        }
                    )

        model = serialized.get("kwargs", {}).get("model_name") or serialized.get("id", [""])[-1]
        context = self._build_context(
            run_id, parent_run_id, "chat_model", serialized, tags, metadata
        )

        decision = self._safe_scan(guard_messages, "input", model, context)
        self._handle_decision(decision, run_id)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if not self._scan_responses:
            return

        text = self._extract_llm_response(response)
        if not text:
            return

        messages = [{"role": "assistant", "content": text}]
        context = self._build_context(run_id, parent_run_id, "llm_response")
        decision = self._safe_scan(messages, "output", context=context)
        self._handle_decision(decision, run_id)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._cleanup_run(run_id)

    # -- Chain callbacks -----------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        run_key = str(run_id) if run_id else "unknown"
        chain_name = serialized.get("id", [""])[-1] if serialized.get("id") else "unknown"
        self._chain_context[run_key] = {
            "chain_name": chain_name,
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": tags,
        }

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._cleanup_run(run_id)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._cleanup_run(run_id)

    # -- Tool callbacks ------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name") or serialized.get("id", [""])[-1]
        messages = [{"role": "user", "content": input_str}]
        context = self._build_context(run_id, parent_run_id, "tool", serialized, tags, metadata)
        context["metadata"] = context.get("metadata", {})
        context["metadata"]["tool_name"] = tool_name

        decision = self._safe_scan(messages, "input", "tool", context)
        self._handle_decision(decision, run_id)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if not self._scan_responses:
            return
        text = str(output) if output else ""
        if not text:
            return
        messages = [{"role": "assistant", "content": text}]
        context = self._build_context(run_id, parent_run_id, "tool_response")
        decision = self._safe_scan(messages, "output", context=context)
        self._handle_decision(decision, run_id)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._cleanup_run(run_id)

    # -- Internal helpers ----------------------------------------------------

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
            logger.warning("Guard API unavailable, allowing %s (fail_open=True)", direction)
            return None

    def _handle_decision(self, decision: GuardDecision | None, run_id: UUID | None) -> None:
        if decision is None:
            return

        if decision.blocked:
            if self._mode == "enforce":
                raise PromptGuardBlockedError(decision)
            logger.warning(
                "[monitor] PromptGuard would block: %s (event=%s, run=%s)",
                decision.threat_type,
                decision.event_id,
                run_id,
            )

        if decision.redacted:
            # The callback observes prompts/responses but cannot rewrite them in
            # flight, so a redact decision can't be applied. In enforce mode we
            # block rather than let unredacted content proceed; monitor warns.
            if self._mode == "enforce":
                logger.error(
                    "PromptGuard: redaction required but the LangChain callback "
                    "cannot rewrite in-flight content; blocking (threat=%s, "
                    "event=%s, run=%s)",
                    decision.threat_type,
                    decision.event_id,
                    run_id,
                )
                raise PromptGuardBlockedError(decision)
            logger.warning(
                "[monitor] PromptGuard would redact content: %s (event=%s, run=%s)",
                decision.threat_type,
                decision.event_id,
                run_id,
            )

    def _build_context(
        self,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        component: str = "unknown",
        serialized: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run_key = str(run_id) if run_id else "unknown"
        chain_info = self._chain_context.get(run_key, {})
        parent_key = str(parent_run_id) if parent_run_id else None
        if parent_key and parent_key in self._chain_context:
            chain_info = self._chain_context[parent_key]

        return {
            "framework": "langchain",
            "chain_name": chain_info.get("chain_name"),
            "session_id": run_key,
            "metadata": {
                "component": component,
                "tags": tags or chain_info.get("tags"),
                **(metadata or {}),
            },
        }

    def _extract_llm_response(self, response: Any) -> str | None:
        try:
            if hasattr(response, "generations") and response.generations:
                texts = []
                for gen_list in response.generations:
                    for gen in gen_list:
                        if hasattr(gen, "text") and gen.text:
                            texts.append(gen.text)
                        elif hasattr(gen, "message") and hasattr(gen.message, "content"):
                            texts.append(str(gen.message.content))
                return "\n".join(texts) if texts else None
        except Exception:
            logger.debug("Failed to extract response text", exc_info=True)
        return None

    def _cleanup_run(self, run_id: UUID | None) -> None:
        run_key = str(run_id) if run_id else None
        if run_key and run_key in self._chain_context:
            del self._chain_context[run_key]


def _require_runnable_lambda() -> Any:
    """Lazily import ``RunnableLambda`` from ``langchain_core``.

    Kept out of module scope so importing ``promptguard.integrations.langchain``
    never requires LangChain to be installed (the callback handler duck-types
    the interface). Only the inline Runnable guard needs the real class.
    """
    try:
        from langchain_core.runnables import RunnableLambda
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "The LangChain Runnable guard requires 'langchain-core', which is "
            "not installed. Install it with: pip install promptguard-sdk[langchain]"
        ) from exc
    return RunnableLambda


# Sentinel distinguishing "no safe redaction" from a legitimately empty result.
_UNREDACTABLE: Any = object()


class PromptGuardRunnable:
    """Inline LCEL guard that scans (and can redact) content mid-chain.

    Unlike :class:`PromptGuardCallbackHandler`, which only *observes* LLM I/O
    and therefore can block but never rewrite it, this guard sits **in** the
    data flow. That lets it honor a ``redact`` decision by returning the
    redacted value, in addition to raising :class:`PromptGuardBlockedError` on a
    ``block`` decision. Compose it into any LCEL chain with ``|``::

        from promptguard.integrations.langchain import PromptGuardRunnable

        guard = PromptGuardRunnable(api_key="pg_live_xxx").as_runnable()
        chain = guard | prompt | llm            # scan the user input
        chain.invoke("user question")

    Use ``direction="output"`` to guard an LLM/parser result at the tail of a
    chain::

        output_guard = PromptGuardRunnable(direction="output").as_runnable()
        chain = prompt | llm | StrOutputParser() | output_guard

    Redaction is applied in place for ``str``, ``dict`` (the first recognized
    text field), and ``list[dict]`` message inputs. For any other value type a
    ``redact`` decision cannot be safely rewritten, so — mirroring the callback
    posture — enforce mode blocks and monitor mode warns and passes through.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        mode: str = "enforce",
        direction: str = "input",
        fail_open: bool = True,
        timeout: float = 10.0,
    ):
        if direction not in ("input", "output"):
            raise ValueError(f"direction must be 'input' or 'output', got {direction!r}")
        resolved_key, resolved_url = resolve_credentials(api_key, base_url)
        validate_mode(mode)
        self._guard = GuardClient(
            api_key=resolved_key,
            base_url=resolved_url,
            timeout=timeout,
        )
        self._mode = mode
        self._direction = direction
        self._fail_open = fail_open
        self._role = "user" if direction == "input" else "assistant"

    def guard_value(self, value: Any) -> Any:
        """Scan ``value``; return it (possibly redacted) or raise on a block.

        This is the plain callable behind :meth:`as_runnable` — usable directly
        (e.g. in a ``RunnableLambda`` you build yourself) without importing
        LangChain.
        """
        messages, kind, dict_key = self._to_messages(value)
        decision = self._safe_scan(messages)
        return self._apply_decision(decision, value, kind, dict_key)

    def as_runnable(self) -> Any:
        """Wrap :meth:`guard_value` in a ``langchain_core`` ``RunnableLambda``."""
        runnable_lambda = _require_runnable_lambda()
        return runnable_lambda(self.guard_value)

    # -- Internal helpers ----------------------------------------------------

    def _to_messages(self, value: Any) -> tuple[list[dict[str, str]], str, str | None]:
        """Normalize ``value`` into Guard messages.

        Returns ``(messages, kind, dict_key)`` where ``kind`` records how to
        re-apply a redaction and ``dict_key`` is the mutated key for dict inputs.
        """
        if isinstance(value, str):
            return [{"role": self._role, "content": value}], "str", None
        if isinstance(value, list) and value and all(isinstance(m, dict) for m in value):
            messages = [
                {"role": str(m.get("role", self._role)), "content": str(m.get("content", ""))}
                for m in value
            ]
            return messages, "messages", None
        if isinstance(value, dict):
            for key in _DICT_TEXT_KEYS:
                if isinstance(value.get(key), str):
                    return [{"role": self._role, "content": value[key]}], "dict", key
        content = getattr(value, "content", None)
        if isinstance(content, str):
            return [{"role": self._role, "content": content}], "other", None
        return [{"role": self._role, "content": str(value)}], "other", None

    def _safe_scan(self, messages: list[dict[str, str]]) -> GuardDecision | None:
        try:
            return self._guard.scan(messages=messages, direction=self._direction)
        except Exception:
            if not self._fail_open:
                raise
            logger.warning("Guard API unavailable, allowing %s (fail_open=True)", self._direction)
            return None

    def _apply_decision(
        self,
        decision: GuardDecision | None,
        value: Any,
        kind: str,
        dict_key: str | None,
    ) -> Any:
        if decision is None or decision.allowed:
            return value

        if decision.blocked:
            if self._mode == "enforce":
                raise PromptGuardBlockedError(decision)
            logger.warning(
                "[monitor] PromptGuard would block: %s (event=%s)",
                decision.threat_type,
                decision.event_id,
            )
            return value

        # redact
        if self._mode != "enforce":
            logger.warning(
                "[monitor] PromptGuard would redact content: %s (event=%s)",
                decision.threat_type,
                decision.event_id,
            )
            return value

        redacted = self._redacted_value(decision, kind, dict_key, value)
        if redacted is not _UNREDACTABLE:
            return redacted

        # Could not rewrite the value in place — fail safe rather than forward
        # the flagged content unredacted.
        logger.error(
            "PromptGuard: redaction required but the %s value could not be "
            "rewritten in place; blocking (threat=%s, event=%s)",
            kind,
            decision.threat_type,
            decision.event_id,
        )
        raise PromptGuardBlockedError(decision)

    def _redacted_value(
        self,
        decision: GuardDecision,
        kind: str,
        dict_key: str | None,
        value: Any,
    ) -> Any:
        redacted_messages = decision.redacted_messages or []
        if kind == "messages":
            # A shorter list would leave trailing messages unredacted.
            if redacted_messages and len(redacted_messages) == len(value):
                return redacted_messages
            return _UNREDACTABLE
        if not redacted_messages:
            return _UNREDACTABLE
        new_content = redacted_messages[0].get("content")
        if not isinstance(new_content, str):
            return _UNREDACTABLE
        if kind == "str":
            return new_content
        if kind == "dict" and dict_key is not None:
            updated = dict(value)
            updated[dict_key] = new_content
            return updated
        return _UNREDACTABLE


# Framework-disambiguating alias. Both the LangChain and LlamaIndex
# integrations historically export a class named ``PromptGuardCallbackHandler``;
# ``LangChainCallbackHandler`` lets callers import both without an alias clash.
# The original name is retained for backwards compatibility.
LangChainCallbackHandler = PromptGuardCallbackHandler

__all__ = [
    "LangChainCallbackHandler",
    "PromptGuardCallbackHandler",
    "PromptGuardRunnable",
]
