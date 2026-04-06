"""
CrewAI integration - guardrail function and tool wrapper.

Usage::

    from crewai import Crew
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
from collections.abc import Callable
from typing import Any

from promptguard._resolve import resolve_credentials
from promptguard.guard import GuardClient, GuardDecision, PromptGuardBlockedError

logger = logging.getLogger("promptguard")


class PromptGuardGuardrail:
    """CrewAI guardrail that scans inputs/outputs via the Guard API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        mode: str = "enforce",
        fail_open: bool = True,
        timeout: float = 10.0,
    ):
        resolved_key, resolved_url = resolve_credentials(api_key, base_url)
        self._guard = GuardClient(
            api_key=resolved_key,
            base_url=resolved_url,
            timeout=timeout,
        )
        self._mode = mode
        self._fail_open = fail_open

    def before_kickoff(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Scan crew inputs before kickoff."""
        messages = self._inputs_to_messages(inputs)
        if not messages:
            return inputs

        context = {"framework": "crewai", "metadata": {"hook": "before_kickoff"}}
        decision = self._scan_and_check(messages, "input", context)

        if decision and decision.redacted and decision.redacted_messages:
            if self._mode == "enforce":
                return self._apply_redaction(inputs, decision.redacted_messages)
            logger.warning(
                "[monitor] PromptGuard would redact crew input: %s",
                decision.threat_type,
            )

        return inputs

    def after_kickoff(self, result: Any) -> Any:
        """Scan crew output after kickoff."""
        text = str(result) if result else ""
        if not text:
            return result

        messages = [{"role": "assistant", "content": text}]
        context = {"framework": "crewai", "metadata": {"hook": "after_kickoff"}}
        self._scan_and_check(messages, "output", context)
        return result

    def scan_task_output(self, output: str, task_name: str = "unknown") -> str:
        """Scan an individual task's output."""
        messages = [{"role": "assistant", "content": output}]
        context = {
            "framework": "crewai",
            "metadata": {"hook": "task_output", "task_name": task_name},
        }
        self._scan_and_check(messages, "output", context)
        return output

    # -- Internal helpers ----------------------------------------------------

    def _scan_and_check(
        self,
        messages: list[dict[str, str]],
        direction: str,
        context: dict[str, Any],
    ) -> GuardDecision | None:
        """Scan content and handle the block decision.  Returns the decision."""
        try:
            decision = self._guard.scan(
                messages=messages,
                direction=direction,
                context=context,
            )
        except Exception:
            if not self._fail_open:
                raise
            logger.warning("Guard API unavailable, allowing request (fail_open=True)")
            return None

        if decision.blocked:
            if self._mode == "enforce":
                raise PromptGuardBlockedError(decision)
            logger.warning(
                "[monitor] PromptGuard would block: %s (event=%s)",
                decision.threat_type,
                decision.event_id,
            )

        return decision

    @staticmethod
    def _inputs_to_messages(inputs: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "user", "content": value}
            for value in inputs.values()
            if isinstance(value, str) and value.strip()
        ]

    @staticmethod
    def _apply_redaction(
        inputs: dict[str, Any],
        redacted_messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        result = dict(inputs)
        idx = 0
        for key in result:
            if (
                isinstance(result[key], str)
                and result[key].strip()
                and idx < len(redacted_messages)
            ):
                result[key] = redacted_messages[idx].get("content", result[key])
                idx += 1
        return result


def secure_tool(
    api_key: str | None = None,
    base_url: str | None = None,
    mode: str = "enforce",
    fail_open: bool = True,
) -> Callable:
    """Decorator to wrap a CrewAI tool's ``_run`` method with PromptGuard."""
    resolved_key, resolved_url = resolve_credentials(api_key, base_url)
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
