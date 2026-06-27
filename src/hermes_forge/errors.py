"""
Forge error hierarchy — all custom exceptions inherit from ForgeError.
"""

from __future__ import annotations


class ForgeError(Exception):
    """Base for all forge exceptions."""

    def __str__(self) -> str:
        return self.message

    @property
    def message(self) -> str:
        return super().__str__() or ""


class ToolCallError(ForgeError):
    """Too many consecutive bad responses or malformed tool calls."""


class ToolExecutionError(ForgeError):
    """Tool execution failed and the model couldn't self-correct."""

    def __init__(self, tool_name: str, cause: Exception | None = None) -> None:
        self.tool_name = tool_name
        self.cause = cause
        super().__init__(
            f"Tool execution error for '{tool_name}': {cause}"
            if cause
            else f"Tool execution error for '{tool_name}'"
        )


class StepEnforcementError(ForgeError):
    """Model repeatedly attempted the terminal tool before completing required steps."""

    def __init__(
        self,
        terminal_tool: str,
        attempts: int,
        pending_steps: list[str],
    ) -> None:
        self.terminal_tool = terminal_tool
        self.attempts = attempts
        self.pending_steps = pending_steps
        super().__init__(
            f"Terminal tool '{terminal_tool}' attempted {attempts} times "
            f"with pending steps: {pending_steps}"
        )


class PrerequisiteError(ForgeError):
    """Tool was called without satisfying its prerequisites."""

    def __init__(
        self,
        tool_name: str,
        violations: int = 0,
        missing_prereqs: list[str] | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.violations = violations
        self.missing_prereqs = missing_prereqs or []
        super().__init__(
            f"Prerequisite violations for '{tool_name}' ({violations}): "
            f"missing: {missing_prereqs}"
        )


class MaxIterationsError(ForgeError):
    """Workflow exceeded max iterations without reaching the terminal tool."""

    def __init__(
        self,
        max_iterations: int,
        completed_steps: list[str],
        pending_steps: list[str],
    ) -> None:
        self.max_iterations = max_iterations
        self.completed_steps = completed_steps
        self.pending_steps = pending_steps
        super().__init__(
            f"Exceeded max iterations ({max_iterations}). "
            f"Completed: {completed_steps}, Pending: {pending_steps}"
        )


class BudgetResolutionError(ForgeError):
    """Could not resolve the context budget from the backend."""

    def __init__(self, cause: Exception | None = None) -> None:
        self.cause = cause
        super().__init__(
            f"Could not resolve context budget: {cause}"
            if cause
            else "Could not resolve context budget"
        )


class BackendError(ForgeError):
    """Backend returned an error response."""

    def __init__(self, status_code: int, body: str = "") -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Backend returned {status_code}: {body[:200]}")
