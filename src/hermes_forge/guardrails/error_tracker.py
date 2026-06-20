"""
ErrorTracker — tracks consecutive retry and tool error budgets.
"""

from __future__ import annotations


class ErrorTracker:
    """Tracks consecutive bad responses and tool execution errors."""

    def __init__(self, max_retries: int = 3, max_tool_errors: int = 2) -> None:
        self.max_retries = max_retries
        self.max_tool_errors = max_tool_errors
        self._retries = 0
        self._errors = 0

    @property
    def retries_exhausted(self) -> bool:
        return self._retries >= self.max_retries

    @property
    def tool_errors_exhausted(self) -> bool:
        return self._errors >= self.max_tool_errors

    def record_retry(self) -> None:
        self._retries += 1

    def record_result(self, success: bool) -> None:
        if success:
            self._errors = 0
        else:
            self._errors += 1

    def reset_retries(self) -> None:
        self._retries = 0

    def reset_errors(self) -> None:
        self._errors = 0
