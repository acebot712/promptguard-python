"""
LangChain integration - PromptGuardCallbackHandler.

Implements LangChain's ``BaseCallbackHandler`` to scan prompts before
LLM calls and responses after, with rich context about chains, tools,
and agent steps.

Usage::

    from promptguard.integrations.langchain import PromptGuardCallbackHandler

    handler = PromptGuardCallbackHandler(api_key="pg_xxx")

    # Attach to a single LLM
    llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])

    # Or use globally with any chain / agent
    chain.invoke({"input": "..."}, config={"callbacks": [handler]})
"""

import logging
import os
from typing import Any
from uuid import UUID

from promptguard.guard import GuardClient, GuardDecision, PromptGuardBlockedError

logger = logging.getLogger("promptguard")


class PromptGuardCallbackHandler:
    """LangChain callback handler that scans LLM I/O via the Guard API.

    This class implements the LangChain ``BaseCallbackHandler`` interface
    without importing LangChain at module level (so the SDK doesn't have
    a hard dependency on it).  LangChain accepts any object with the right
    methods as a callback handler.

    Parameters
    ----------
    api_key:
        PromptGuard API key.  Falls back to ``PROMPTGUARD_API_KEY``.
    base_url:
        Guard API base URL.
    mode:
        ``"enforce"`` to block, ``"monitor"`` to log only.
    scan_responses:
        Whether to also scan LLM responses (output direction).
    fail_open:
        If True, allow the call when the Guard API is unreachable.
    timeout:
        HTTP timeout for guard API calls.
    """

    # LangChain checks these flags to decide which callbacks to call.
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
        resolved_key = api_key or os.environ.get("PROMPTGUARD_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "PromptGuard API key required. Pass api_key= or set "
                "PROMPTGUARD_API_KEY environment variable."
            )

        resolved_url = (
            base_url
            or os.environ.get("PROMPTGUARD_BASE_URL")
            or "https://api.promptguard.co/api/v1"
        )

        self._guard = GuardClient(
            api_key=resolved_key,
            base_url=resolved_url,
            timeout=timeout,
        )
        self._mode = mode
        self._scan_responses = scan_responses
        self._fail_open = fail_open

        # Track chain context per run_id
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
        """Called when an LLM starts running. Scans the prompts."""
        messages = [{"role": "user", "content": p} for p in prompts]
        model = serialized.get("kwargs", {}).get("model_name") or serialized.get("id", [""])[-1]
        context = self._build_context(run_id, parent_run_id, "llm", serialized, tags, metadata)

        decision = self._scan_input(messages, model, context)
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
        """Called when a chat model starts. Scans the messages."""
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
            run_id,
            parent_run_id,
            "chat_model",
            serialized,
            tags,
            metadata,
        )

        decision = self._scan_input(guard_messages, model, context)
        self._handle_decision(decision, run_id)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM finishes. Optionally scans the response."""
        if not self._scan_responses:
            return

        text = self._extract_llm_response(response)
        if not text:
            return

        messages = [{"role": "assistant", "content": text}]
        context = self._build_context(run_id, parent_run_id, "llm_response")
        decision = self._scan_output(messages, context)
        self._handle_decision(decision, run_id)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called on LLM error - clean up context."""
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
        """Track chain context for richer threat detection."""
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
        """Clean up chain context."""
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
        """Scan tool inputs for injection attempts."""
        tool_name = serialized.get("name") or serialized.get("id", [""])[-1]
        messages = [{"role": "user", "content": input_str}]
        context = self._build_context(run_id, parent_run_id, "tool", serialized, tags, metadata)
        context["metadata"] = context.get("metadata", {})
        context["metadata"]["tool_name"] = tool_name

        decision = self._scan_input(messages, "tool", context)
        self._handle_decision(decision, run_id)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Optionally scan tool outputs."""
        if not self._scan_responses:
            return
        text = str(output) if output else ""
        if not text:
            return
        messages = [{"role": "assistant", "content": text}]
        context = self._build_context(run_id, parent_run_id, "tool_response")
        decision = self._scan_output(messages, context)
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

    def _scan_input(
        self,
        messages: list[dict[str, str]],
        model: str | None,
        context: dict[str, Any],
    ) -> GuardDecision | None:
        try:
            return self._guard.scan(
                messages=messages,
                direction="input",
                model=model,
                context=context,
            )
        except Exception:
            if not self._fail_open:
                raise
            logger.warning("Guard API unavailable, allowing request (fail_open=True)")
            return None

    def _scan_output(
        self,
        messages: list[dict[str, str]],
        context: dict[str, Any],
    ) -> GuardDecision | None:
        try:
            return self._guard.scan(
                messages=messages,
                direction="output",
                context=context,
            )
        except Exception:
            if not self._fail_open:
                raise
            logger.warning("Guard API unavailable, allowing response (fail_open=True)")
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
            logger.info(
                "PromptGuard redacted content: %s (event=%s, run=%s)",
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

        context: dict[str, Any] = {
            "framework": "langchain",
            "chain_name": chain_info.get("chain_name"),
            "session_id": run_key,
            "metadata": {
                "component": component,
                "tags": tags or chain_info.get("tags"),
                **(metadata or {}),
            },
        }
        return context

    def _extract_llm_response(self, response: Any) -> str | None:
        """Extract text from a LangChain LLMResult."""
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
