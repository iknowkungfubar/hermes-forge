"""
Compaction strategies for context window management.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


def _estimate_tokens(messages: list) -> int:
    """Estimate token count using tiktoken if available, fall back to char-based."""
    try:
        import tiktoken  # type: ignore[import-not-found]

        enc = tiktoken.get_encoding("cl100k_base")
        total = 0
        for m in messages:
            content = m.content if hasattr(m, "content") else str(m)
            total += len(enc.encode(content))
        return total
    except ImportError:
        return sum(len(m.content) for m in messages) // 4


class CompactStrategy(ABC):
    """Interface for context compaction strategies."""

    @abstractmethod
    def compact(
        self,
        messages: list,
        budget_tokens: int,
        *,
        step_hint: str = "",
    ) -> tuple[list, int]: ...


class NoCompact(CompactStrategy):
    """Passthrough — returns messages unchanged."""

    def compact(
        self,
        messages: list,
        budget_tokens: int,
        *,
        step_hint: str = "",
    ) -> tuple[list, int]:
        return list(messages), 0


class SlidingWindowCompact(CompactStrategy):
    """Keeps system prompt, user input, and last N iterations."""

    def __init__(self, keep_recent: int, compact_threshold: float = 0.75) -> None:
        self.keep_recent = keep_recent
        self.compact_threshold = compact_threshold

    def compact(
        self,
        messages: list,
        budget_tokens: int,
        *,
        step_hint: str = "",
    ) -> tuple[list, int]:
        trigger = int(budget_tokens * self.compact_threshold)
        if _estimate_tokens(messages) < trigger:
            return list(messages), 0
        eligible_end = self._find_eligible_end(messages, self.keep_recent)
        if eligible_end <= 2:
            return list(messages), 1
        return [messages[0], messages[1]] + messages[eligible_end:], 1

    @staticmethod
    def _find_eligible_end(messages: list, keep_recent: int) -> int:
        seen_steps: list[int] = []
        for m in messages[2:]:
            si = getattr(m, "metadata", None)
            step_idx = getattr(si, "step_index", None) if si else None
            if step_idx is not None and (not seen_steps or seen_steps[-1] != step_idx):
                seen_steps.append(step_idx)
        if len(seen_steps) <= keep_recent:
            return 2
        cutoff_step = seen_steps[-keep_recent]
        for i in range(2, len(messages)):
            si = getattr(messages[i], "metadata", None)
            step_idx = getattr(si, "step_index", None) if si else None
            if step_idx is not None and step_idx >= cutoff_step:
                return i
        return len(messages)


class TieredCompact(CompactStrategy):
    """Three-phase compaction with explicit priority order.

    Phase 1: Drop nudges/retries, truncate tool_results
    Phase 2: Drop tool_results entirely
    Phase 3: Drop reasoning and text responses (tool_call skeleton only)
    """

    TRUNCATE_CHARS = 200

    def __init__(
        self,
        keep_recent: int = 2,
        compact_threshold: float = 0.75,
        phase_thresholds: tuple[float, float, float] | None = None,
    ) -> None:
        self.keep_recent = keep_recent
        if phase_thresholds is not None:
            self._phase_triggers = phase_thresholds
        else:
            self._phase_triggers = (
                compact_threshold,
                compact_threshold,
                compact_threshold,
            )

    @staticmethod
    def _find_eligible_end(messages: list, keep_recent: int) -> int:
        seen_steps: list[int] = []
        for m in messages[2:]:
            si = getattr(m, "metadata", None)
            step_idx = getattr(si, "step_index", None) if si else None
            if step_idx is not None and (not seen_steps or seen_steps[-1] != step_idx):
                seen_steps.append(step_idx)
        if len(seen_steps) <= keep_recent:
            return 2
        cutoff_step = seen_steps[-keep_recent]
        for i in range(2, len(messages)):
            si = getattr(messages[i], "metadata", None)
            step_idx = getattr(si, "step_index", None) if si else None
            if step_idx is not None and step_idx >= cutoff_step:
                return i
        return len(messages)

    def compact(
        self,
        messages: list,
        budget_tokens: int,
        *,
        step_hint: str = "",
    ) -> tuple[list, int]:
        tokens = _estimate_tokens(messages)
        t1 = int(budget_tokens * self._phase_triggers[0])
        t2 = int(budget_tokens * self._phase_triggers[1])
        t3 = int(budget_tokens * self._phase_triggers[2])

        if tokens < t1:
            return list(messages), 0

        eligible_end = self._find_eligible_end(messages, self.keep_recent)

        result = self._phase1(messages, eligible_end)
        if _estimate_tokens(result) < t2:
            return result, 1

        result = self._phase2(messages, eligible_end)
        if _estimate_tokens(result) < t3:
            return result, 2

        result = self._phase3(messages, eligible_end)
        return result, 3

    def _is_type(self, msg, *types: str) -> bool:
        meta = getattr(msg, "metadata", None)
        if meta is None:
            return False
        msg_type = getattr(meta, "type", None)
        if msg_type is None:
            return False
        return msg_type.value in types

    def _phase1(self, messages: list, eligible_end: int) -> list:
        result: list = []
        from hermes_forge.core.messages import MessageType

        drop_types = {
            MessageType.STEP_NUDGE.value,
            MessageType.PREREQUISITE_NUDGE.value,
            MessageType.RETRY_NUDGE.value,
        }
        for i, msg in enumerate(messages):
            if 2 <= i < eligible_end:
                if self._is_type(msg, *drop_types):
                    continue
                if self._is_type(msg, MessageType.TOOL_RESULT.value):
                    if len(msg.content) > self.TRUNCATE_CHARS:
                        from copy import deepcopy

                        truncated = deepcopy(msg)
                        truncated.content = (
                            msg.content[: self.TRUNCATE_CHARS]
                            + f"\n[Truncated — {len(msg.content) - self.TRUNCATE_CHARS} chars removed]"
                        )
                        result.append(truncated)
                        continue
            result.append(msg)
        return result

    def _phase2(self, messages: list, eligible_end: int) -> list:
        result: list = []
        from hermes_forge.core.messages import MessageType

        drop_types = {
            MessageType.STEP_NUDGE.value,
            MessageType.PREREQUISITE_NUDGE.value,
            MessageType.RETRY_NUDGE.value,
            MessageType.TOOL_RESULT.value,
        }
        for i, msg in enumerate(messages):
            if 2 <= i < eligible_end:
                if self._is_type(msg, *drop_types):
                    continue
            result.append(msg)
        return result

    def _phase3(self, messages: list, eligible_end: int) -> list:
        result: list = []
        from hermes_forge.core.messages import MessageType

        drop_types = {
            MessageType.STEP_NUDGE.value,
            MessageType.PREREQUISITE_NUDGE.value,
            MessageType.RETRY_NUDGE.value,
            MessageType.TOOL_RESULT.value,
            MessageType.REASONING.value,
            MessageType.TEXT_RESPONSE.value,
        }
        for i, msg in enumerate(messages):
            if 2 <= i < eligible_end:
                if self._is_type(msg, *drop_types):
                    continue
            result.append(msg)
        return result
