"""
Message types for the forge agentic loop.

These are the canonical message representations used throughout the
guardrail pipeline. All backends (Ollama, llama.cpp, vLLM, Anthropic)
are normalized to these types before guardrails are applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MessageType(str, Enum):
    SYSTEM_PROMPT = "system_prompt"
    USER_INPUT = "user_input"
    TEXT_RESPONSE = "text_response"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RETRY_NUDGE = "retry_nudge"
    STEP_NUDGE = "step_nudge"
    PREREQUISITE_NUDGE = "prerequisite_nudge"


@dataclass
class ToolCallInfo:
    """Metadata for a single tool call within a Message."""

    name: str
    call_id: str
    args: str  # JSON-encoded arguments


@dataclass
class MessageMeta:
    """Metadata attached to each Message."""

    type: MessageType = MessageType.TEXT_RESPONSE
    step_index: int | None = None


@dataclass
class Message:
    """Canonical message in the forge conversation format."""

    role: MessageRole
    content: str
    metadata: MessageMeta = field(default_factory=MessageMeta)
    tool_calls: list[ToolCallInfo] | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
