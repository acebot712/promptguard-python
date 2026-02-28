"""
CrewAI integration — guardrail function and tool wrapper.

CrewAI supports ``before_kickoff`` / ``after_kickoff`` hooks on the Crew
class.  This module provides:

1.  ``promptguard_guardrail`` — a guardrail function for use with
    CrewAI's ``@before_kickoff`` / ``@after_kickoff`` hooks.
2.  ``secure_tool`` — a decorator that wraps any CrewAI tool's
    ``_run`` method with PromptGuard validation.

Usage::

    from crewai import Crew, Agent, Task
    from promptguard.integrations.crewai import PromptGuardGuardrail

    pg = PromptGuardGuardrail(api_key="pg_xxx")

    crew = Crew(
        agents=[...],
        tasks=[...],
        before_kickoff=pg.before_kickoff,
        after_kickoff=pg.after_kickoff,
    )

    # Or wrap individual tools:
    from promptguard.integrations.crewai import secure_tool

    @secure_tool(api_key="pg_xxx")
    class MyTool(BaseTool):
        ...
"""

import functools
import logging
import os
from collections.abc import Callable
from typing import Any

from promptguard.guard import GuardClient, PromptGuardBlockedError

logger = logging.getLogger("promptguard")


class PromptGuardGuardrail:
    """CrewAI guardrail that scans inputs/outputs via the Guard API.

    Parameters
    ----------
    api_key:
        PromptGuard API key.
    base_url:
        Guard API base URL.
    mode:
        ``"enforce"`` to block, ``"monitor"`` to log only.
    fail_open:
        Allow execution when Guard API is unreachable.
    timeout:
        HTTP timeout for guard API calls.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        mode: str = "enforce",
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
        self._fail_open = fail_open

    def before_kickoff(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Scan crew inputs before kickoff.

        Suitable for use as ``Crew(before_kickoff=pg.before_kickoff)``.
        Scans all string values in the inputs dict.
        """
        messages = self._inputs_to_messages(inputs)
        if not messages:
            return inputs

        context = {
            "framework": "crewai",
            "metadata": {"hook": "before_kickoff"},
        }

        try:
            decision = self._guard.scan(
                messages=messages,
                direction="input",
                context=context,
            )
        except Exception:
            if not self._fail_open:
                raise
            logger.warning("Guard API unavailable, allowing crew kickoff (fail_open=True)")
            return inputs

        if decision.blocked:
            if self._mode == "enforce":
                raise PromptGuardBlockedError(decision)
            else:
                logger.warning(
                    "[monitor] PromptGuard would block crew input: %s (event=%s)",
                    decision.threat_type,
                    decision.event_id,
                )

        if decision.redacted and decision.redacted_messages:
            if self._mode == "enforce":
                return self._apply_redaction(inputs, decision.redacted_messages)
            else:
                logger.warning(
                    "[monitor] PromptGuard would redact crew input: %s",
                    decision.threat_type,
                )

        return inputs

    def after_kickoff(self, result: Any) -> Any:
        """Scan crew output after kickoff.

        Suitable for use as ``Crew(after_kickoff=pg.after_kickoff)``.
        """
        text = str(result) if result else ""
        if not text:
            return result

        messages = [{"role": "assistant", "content": text}]
        context = {
            "framework": "crewai",
            "metadata": {"hook": "after_kickoff"},
        }

        try:
            decision = self._guard.scan(
                messages=messages,
                direction="output",
                context=context,
            )
        except Exception:
            if not self._fail_open:
                raise
            logger.warning("Guard API unavailable, allowing crew output (fail_open=True)")
            return result

        if decision.blocked:
            if self._mode == "enforce":
                raise PromptGuardBlockedError(decision)
            else:
                logger.warning(
                    "[monitor] PromptGuard would block crew output: %s (event=%s)",
                    decision.threat_type,
                    decision.event_id,
                )

        return result

    def scan_task_output(self, output: str, task_name: str = "unknown") -> str:
        """Scan an individual task's output.

        Can be used as a CrewAI task ``output_validator`` or called
        manually between tasks.
        """
        messages = [{"role": "assistant", "content": output}]
        context = {
            "framework": "crewai",
            "metadata": {"hook": "task_output", "task_name": task_name},
        }

        try:
            decision = self._guard.scan(
                messages=messages,
                direction="output",
                context=context,
            )
        except Exception:
            if not self._fail_open:
                raise
            return output

        if decision.blocked:
            if self._mode == "enforce":
                raise PromptGuardBlockedError(decision)
            else:
                logger.warning(
                    "[monitor] PromptGuard would block task output: %s", decision.threat_type
                )

        return output

    @staticmethod
    def _inputs_to_messages(inputs: dict[str, Any]) -> list[dict[str, str]]:
        messages = []
        for _key, value in inputs.items():
            if isinstance(value, str) and value.strip():
                messages.append({"role": "user", "content": value})
        return messages

    @staticmethod
    def _apply_redaction(
        inputs: dict[str, Any],
        redacted_messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        result = dict(inputs)
        idx = 0
        for key in result:
            if isinstance(result[key], str) and result[key].strip():
                if idx < len(redacted_messages):
                    result[key] = redacted_messages[idx].get("content", result[key])
                    idx += 1
        return result


def secure_tool(
    api_key: str | None = None,
    base_url: str | None = None,
    mode: str = "enforce",
    fail_open: bool = True,
) -> Callable:
    """Decorator to wrap a CrewAI tool's ``_run`` method with PromptGuard.

    Usage::

        @secure_tool(api_key="pg_xxx")
        class SearchTool(BaseTool):
            name = "search"
            description = "Search the web"

            def _run(self, query: str) -> str:
                ...
    """
    resolved_key = api_key or os.environ.get("PROMPTGUARD_API_KEY", "")
    resolved_url = (
        base_url or os.environ.get("PROMPTGUARD_BASE_URL") or "https://api.promptguard.co/api/v1"
    )

    guard = GuardClient(api_key=resolved_key, base_url=resolved_url)

    def decorator(cls):
        original_run = cls._run

        @functools.wraps(original_run)
        def wrapped_run(self, *args, **kwargs):
            input_text = str(args[0]) if args else str(kwargs)
            tool_name = getattr(self, "name", cls.__name__)

            messages = [{"role": "user", "content": input_text}]
            context = {
                "framework": "crewai",
                "metadata": {"tool_name": tool_name, "hook": "tool_run"},
            }

            try:
                decision = guard.scan(
                    messages=messages,
                    direction="input",
                    context=context,
                )
                if decision.blocked:
                    if mode == "enforce":
                        raise PromptGuardBlockedError(decision)
                    else:
                        logger.warning(
                            "[monitor] PromptGuard would block tool %s: %s",
                            tool_name,
                            decision.threat_type,
                        )
            except PromptGuardBlockedError:
                raise
            except Exception:
                if not fail_open:
                    raise

            return original_run(self, *args, **kwargs)

        cls._run = wrapped_run
        return cls

    return decorator
