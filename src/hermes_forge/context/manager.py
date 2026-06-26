"""
Context management: ContextManager and compaction event types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CompactEvent:
    """Event fired when context compaction occurs."""

    phase: int
    before_tokens: int
    after_tokens: int
    budget: int
    message_count_before: int
    message_count_after: int


class ContextManager:
    """Manages context budget and triggers compaction.

    Delegates compaction strategy to a CompactStrategy instance.
    """

    def __init__(
        self,
        strategy: Any = None,  # CompactStrategy
        budget_tokens: int = 8192,
        on_compact: Any | None = None,
        context_thresholds: list[float] | None = None,
        on_context_threshold: Any | None = None,
    ) -> None:
        self._strategy = strategy
        self.budget_tokens = budget_tokens
        self._on_compact = on_compact
        self._on_context_threshold = on_context_threshold
        self._total_compactions = 0

    @property
    def strategy(self) -> Any:
        return self._strategy

    @property
    def total_compactions(self) -> int:
        return self._total_compactions

    def should_compact(self, messages: list) -> tuple[bool, int]:
        """Check if compaction is needed based on estimated tokens vs budget."""
        if self._strategy is None:
            return False, 0
        tokens = sum(len(m.content) for m in messages) // 4
        return tokens >= int(self.budget_tokens * 0.75), tokens

    def compact(self, messages: list, step_hint: str = "") -> list:
        """Compact messages if they exceed the threshold."""
        if self._strategy is None:
            return messages

        before_tokens = sum(len(m.content) for m in messages) // 4
        before_count = len(messages)

        compacted, phase = self._strategy.compact(
            messages, self.budget_tokens, step_hint=step_hint
        )

        if phase > 0:
            self._total_compactions += 1
            if self._on_compact:
                self._on_compact(
                    CompactEvent(
                        phase=phase,
                        before_tokens=before_tokens,
                        after_tokens=sum(len(m.content) for m in compacted) // 4,
                        budget=self.budget_tokens,
                        message_count_before=before_count,
                        message_count_after=len(compacted),
                    )
                )

        return compacted
