"""
LLM Client protocol and base types.

All backend clients (Ollama, Llamafile, vLLM, Anthropic, OpenAI-compat)
implement the LLMClient protocol. They normalize backend-specific responses
to forge's canonical Message types so guardrails apply uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ChunkType(str, Enum):
    TEXT = "text"
    TOOL_CALL = "tool_call"
    REASONING = "reasoning"
    DONE = "done"
    ERROR = "error"


@dataclass
class StreamChunk:
    """A single chunk from a streaming LLM response."""
    type: ChunkType
    content: str = ""
    tool_name: str | None = None
    tool_args: str | None = None  # JSON-encoded partial args
    finish_reason: str | None = None


@dataclass
class TokenUsage:
    """Token usage information from an LLM response."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMClient(ABC):
    """Protocol for LLM backend clients."""

    @abstractmethod
    async def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], TokenUsage]:
        """Send messages to the LLM and return tool calls or text response.

        Args:
            messages: List of message dicts in OpenAI format.
            tools: Optional list of tool definitions in OpenAI format.

        Returns:
            Tuple of (response_data, token_usage).
            response_data is a list of tool call dicts, or a single
            message dict with content field for text responses.
        """
        ...

    @abstractmethod
    async def send_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        """Stream a response from the LLM.

        Yields StreamChunk objects as they arrive.
        Must be implemented as an async generator.
        """
        ...
        # To make it an abstract async generator, we use a sentinel yield
        # that subclasses must override with their own implementation.
        # This pattern works around Python's inability to define abstract
        # async generators directly.
        if False:
            yield StreamChunk(type=ChunkType.TEXT)  # pragma: no cover

    @abstractmethod
    async def get_context_length(self) -> int | None:
        """Get the context length reported by the backend, if available."""
        ...

    @abstractmethod
    async def aclose(self) -> None:
        """Close the client and release resources."""
        ...


# Need AsyncIterator
