"""
Guardrails — bundled middleware facade for foreign orchestration loops.

Two-method API: check(response) + record(executed_tools).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from hermes_forge.core.workflow import LLMResponse, ToolCall
from hermes_forge.guardrails.error_tracker import ErrorTracker
from hermes_forge.guardrails.nudge import TOOL_CHANNEL_KINDS, TOOL_ERROR_KINDS, Nudge
from hermes_forge.guardrails.response_validator import ResponseValidator
from hermes_forge.guardrails.step_enforcer import StepEnforcer


@dataclass(frozen=True)
class CheckResult:
    """Result of checking an LLM response against all guardrails."""

    action: Literal["execute", "retry", "tool_error", "step_blocked", "fatal"]
    tool_calls: list[ToolCall] | None = None
    nudge: Nudge | None = None
    reason: str | None = None


class Guardrails:
    """Bundled guardrail middleware for foreign orchestration loops.

    Wraps ResponseValidator, StepEnforcer, and ErrorTracker into a
    two-method API. Use check() after each LLM response and record()
    after executing tools.
    """

    def __init__(
        self,
        tool_names: list[str],
        terminal_tool: str | frozenset[str],
        required_steps: list[str] | None = None,
        max_retries: int = 3,
        max_tool_errors: int = 2,
        rescue_enabled: bool = True,
        max_premature_attempts: int = 3,
        retry_nudge: Callable[[str], str] | None = None,
    ) -> None:
        self._validator = ResponseValidator(
            tool_names=tool_names,
            rescue_enabled=rescue_enabled,
            retry_nudge_fn=retry_nudge,
        )
        if isinstance(terminal_tool, str):
            terminal_tools = frozenset([terminal_tool])
        else:
            terminal_tools = terminal_tool
        self._enforcer = StepEnforcer(
            required_steps=required_steps or [],
            terminal_tools=terminal_tools,
            max_premature_attempts=max_premature_attempts,
        )
        self._errors = ErrorTracker(
            max_retries=max_retries,
            max_tool_errors=max_tool_errors,
        )

    def check(self, response: LLMResponse) -> CheckResult:
        """Check an LLM response against all guardrails."""
        validation = self._validator.validate(response)
        if validation.needs_retry:
            nudge = validation.nudge
            kind = nudge.kind if nudge else ""
            if kind in TOOL_ERROR_KINDS:
                self._errors.record_result(success=False)
                if self._errors.tool_errors_exhausted:
                    return CheckResult(
                        action="fatal",
                        reason="too many consecutive tool-argument errors",
                    )
            else:
                self._errors.record_retry()
                if self._errors.retries_exhausted:
                    return CheckResult(
                        action="fatal", reason="too many consecutive bad responses"
                    )
            action: Literal["tool_error", "retry"] = "tool_error" if kind in TOOL_CHANNEL_KINDS else "retry"
            return CheckResult(action=action, nudge=nudge)

        self._errors.reset_retries()
        if validation.tool_calls is None:
            return CheckResult(action="retry", nudge=Nudge("retry", "empty tool calls"))
        step_check = self._enforcer.check(validation.tool_calls)
        if step_check.needs_nudge:
            if self._enforcer.premature_exhausted:
                return CheckResult(
                    action="fatal", reason="model repeatedly skipped required steps"
                )
            return CheckResult(action="step_blocked", nudge=step_check.nudge)

        return CheckResult(action="execute", tool_calls=validation.tool_calls)

    def record(self, executed: list[str | tuple[str, dict]]) -> bool:
        """Record which tools were successfully executed.

        Returns True if the terminal tool was reached and all required steps
        are satisfied (workflow is done).
        """
        for entry in executed:
            if isinstance(entry, tuple):
                name, args = entry
                self._enforcer.record(name, args)
            else:
                self._enforcer.record(entry, {})
        self._errors.reset_errors()
        self._enforcer.reset_premature()
        terminal_reached = any(
            (entry if isinstance(entry, str) else entry[0])
            in self._enforcer.terminal_tools
            for entry in executed
        )
        return self._enforcer.is_satisfied() and terminal_reached
