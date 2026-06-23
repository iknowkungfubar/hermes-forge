"""Reasoning replay policy shared by runner and proxy.

Controls how reasoning/thinking content from models that support it
(DeepSeek R1, Qwen with thinking mode, etc.) is handled across turns.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

ReasoningReplay = Literal["full", "keep-last", "none"]
REASONING_REPLAY_CHOICES: tuple[ReasoningReplay, ...] = ("full", "keep-last", "none")
DEFAULT_REASONING_REPLAY: ReasoningReplay = "none"


def validate_reasoning_replay(value: str) -> ReasoningReplay:
    """Validate and normalize a reasoning replay policy."""
    if value not in REASONING_REPLAY_CHOICES:
        choices = ", ".join(REASONING_REPLAY_CHOICES)
        raise ValueError(f"reasoning_replay must be one of: {choices}")
    return value  # type: ignore[return-value]


REASONING_MESSAGE_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")


def filter_openai_reasoning_messages(
    messages: list[dict[str, Any]],
    reasoning_replay: ReasoningReplay = DEFAULT_REASONING_REPLAY,
) -> list[dict[str, Any]]:
    """Copy raw OpenAI messages and apply the reasoning replay policy.

    - ``none``: strip reasoning fields entirely (most token-efficient).
    - ``keep-last``: keep reasoning from the most recent assistant turn, strip older ones.
    - ``full``: preserve all reasoning fields (most transparent, largest context).

    Returns a deep copy — the original list is not mutated.
    """
    if reasoning_replay == "none":
        return _strip_all_reasoning(messages)

    result = deepcopy(messages)
    if reasoning_replay == "keep-last":
        _strip_all_but_last_reasoning(result)

    # For "full" we just return the deep copy unchanged
    return result


def _strip_all_reasoning(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy with all reasoning fields removed."""
    result = []
    for msg in messages:
        copy = dict(msg)
        for field in REASONING_MESSAGE_FIELDS:
            copy.pop(field, None)
        result.append(copy)
    return result


def _strip_all_but_last_reasoning(messages: list[dict[str, Any]]) -> None:
    """Remove reasoning fields from all but the last assistant message, in-place."""
    last_assistant_idx = -1
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            for field in REASONING_MESSAGE_FIELDS:
                if field in msg:
                    last_assistant_idx = i
                    break

    for i, msg in enumerate(messages):
        if i != last_assistant_idx and msg.get("role") == "assistant":
            for field in REASONING_MESSAGE_FIELDS:
                msg.pop(field, None)
