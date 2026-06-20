"""
StepTracker — tracks completed steps and prerequisite satisfaction.
"""

from __future__ import annotations

from typing import Any


class StepTracker:
    """Tracks completed tool calls and prerequisite satisfaction."""

    def __init__(self) -> None:
        self._completed: list[tuple[str, dict[str, Any]]] = []

    def record(self, tool_name: str, args: dict[str, Any]) -> None:
        self._completed.append((tool_name, args))

    @property
    def completed_tools(self) -> list[str]:
        return [name for name, _ in self._completed]

    @property
    def completed(self) -> list[tuple[str, dict[str, Any]]]:
        return list(self._completed)

    def was_called(self, tool_name: str) -> bool:
        return any(tool_name == name for name, _ in self._completed)

    def check_prerequisites(
        self,
        tool_name: str,
        args: dict[str, Any],
        prerequisites: list[str | dict[str, str]],
    ) -> "PrerequisiteResult":
        """Check if all prerequisites for a tool are satisfied."""
        missing: list[str] = []
        for prereq in prerequisites:
            if isinstance(prereq, str):
                if not self.was_called(prereq):
                    missing.append(prereq)
            elif isinstance(prereq, dict):
                prereq_tool = prereq.get("tool", "")
                match_arg = prereq.get("match_arg")
                if match_arg:
                    match_value = args.get(match_arg)
                    if not any(
                        n == prereq_tool and a.get(match_arg) == match_value
                        for n, a in self._completed
                    ):
                        missing.append(f"{prereq_tool}({match_arg}={match_value})")
                else:
                    if not self.was_called(prereq_tool):
                        missing.append(prereq_tool)
        return PrerequisiteResult(satisfied=len(missing) == 0, missing=missing)


class PrerequisiteResult:
    """Result of a prerequisite check."""

    def __init__(self, satisfied: bool, missing: list[str]) -> None:
        self.satisfied = satisfied
        self.missing = missing
