"""
StepEnforcer — enforces required steps and prerequisite ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hermes_forge.core.steps import StepTracker
from hermes_forge.core.workflow import ToolCall
from hermes_forge.guardrails.nudge import Nudge


@dataclass
class StepCheck:
    """Result of check() — does the model need a nudge?"""
    needs_nudge: bool = False
    nudge: Nudge | None = None


class StepEnforcer:
    """Enforces required steps and prerequisites.

    - required_steps: tools that must be called before terminal
    - terminal_tools: tools that end the workflow
    - tool_prerequisites: per-tool prerequisites (arg-matched)
    """

    def __init__(
        self,
        required_steps: list[str],
        terminal_tools: frozenset[str],
        tool_prerequisites: dict[str, list[str | dict[str, str]]] | None = None,
        max_premature_attempts: int = 3,
    ) -> None:
        self.required_steps = required_steps
        self.terminal_tools = terminal_tools
        self._tool_prerequisites = tool_prerequisites or {}
        self._max_premature = max_premature_attempts
        self._tracker = StepTracker()
        self._premature_attempts = 0
        self._prereq_violations: int = 0

    @property
    def completed_steps(self) -> list[str]:
        return self._tracker.completed_tools

    @property
    def premature_exhausted(self) -> bool:
        return self._premature_attempts >= self._max_premature

    @property
    def premature_attempts(self) -> int:
        return self._premature_attempts

    @property
    def prereq_violations(self) -> int:
        return self._prereq_violations

    def pending(self) -> list[str]:
        """Return required steps not yet completed."""
        return [s for s in self.required_steps if not self._tracker.was_called(s)]

    def is_satisfied(self) -> bool:
        """All required steps completed?"""
        return len(self.pending()) == 0

    def summary_hint(self) -> str:
        """Return a one-line hint about remaining steps for the system prompt."""
        pending = self.pending()
        if not pending:
            return "All required steps completed. You may call the terminal tool."
        return f"Required steps remaining: {', '.join(pending)}"

    def check(self, tool_calls: list[ToolCall]) -> StepCheck:
        """Check if the model is skipping required steps or going terminal too early.

        Returns a StepCheck with needs_nudge=True if:
        - Any terminal tool is called before all required steps are done
        """
        for tc in tool_calls:
            if tc.tool in self.terminal_tools and not self.is_satisfied():
                self._premature_attempts += 1
                pending = self.pending()
                return StepCheck(
                    needs_nudge=True,
                    nudge=Nudge(
                        "step_enforcement",
                        f"Cannot call '{tc.tool}' yet. "
                        f"Complete these steps first: {', '.join(pending)}. "
                        f"Progress: {', '.join(self._tracker.completed_tools)}.",
                    ),
                )
        return StepCheck()

    def check_prerequisites(self, tool_calls: list[ToolCall]) -> StepCheck:
        """Check if any tool call violates prerequisites."""
        for tc in tool_calls:
            prereqs = self._tool_prerequisites.get(tc.tool, [])
            if prereqs:
                result = self._tracker.check_prerequisites(tc.tool, tc.args, prereqs)
                if not result.satisfied:
                    self._prereq_violations += 1
                    return StepCheck(
                        needs_nudge=True,
                        nudge=Nudge(
                            "prerequisite_skip",
                            f"Cannot call '{tc.tool}' yet. "
                            f"Prerequisites not met: {', '.join(result.missing)}.",
                        ),
                    )
        return StepCheck()

    def record(self, tool_name: str, args: dict[str, Any]) -> None:
        """Record a successful tool execution."""
        self._tracker.record(tool_name, args)

    def reset_premature(self) -> None:
        self._premature_attempts = 0

    def reset_prereq_violations(self) -> None:
        self._prereq_violations = 0
