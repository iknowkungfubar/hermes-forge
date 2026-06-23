"""Inference loop — compact, fold, serialize, send, validate, retry.

Extracted so both the WorkflowRunner and the Proxy can share the same
input-processing and validation logic. This is the "front half" of the
agentic loop: everything up to and including getting a clean response
from the LLM.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from hermes_forge.core.messages import Message, MessageMeta, MessageRole, MessageType, ToolCallInfo
from hermes_forge.core.reasoning import (
    DEFAULT_REASONING_REPLAY,
    ReasoningReplay,
    filter_openai_reasoning_messages,
    validate_reasoning_replay,
)
from hermes_forge.core.workflow import ToolCall, ToolSpec
from hermes_forge.errors import ToolCallError
from hermes_forge.guardrails.response_validator import rescue_tool_call


@dataclass
class InferenceResult:
    """The validated result of a single inference pass."""

    tool_calls: list[ToolCall]
    text: str | None = None
    token_usage: dict[str, int] | None = None
    needs_retry: bool = False
    retry_reason: str | None = None


def _tool_names(tools: dict[str, ToolSpec] | list[str] | None) -> set[str]:
    """Convert various tool representations to a set of tool names."""
    if tools is None:
        return set()
    if isinstance(tools, dict):
        return set(tools.keys())
    return set(tools)


def run_inference(
    response_text: str,
    tools: dict[str, ToolSpec] | list[str] | None = None,
    reasoning_replay: ReasoningReplay = DEFAULT_REASONING_REPLAY,
) -> InferenceResult:
    """Process an LLM response through the forge guardrail pipeline.

    Given raw text from an LLM response:
    1. Attempt to parse as JSON tool call(s)
    2. If that fails, try rescue parsing (code fences, Qwen XML, Mistral, etc.)
    3. If rescue fails, return needs_retry=True with the reason

    Returns InferenceResult with either tool_calls or text.
    """
    valid_tools = _tool_names(tools)

    # Step 1: Try direct JSON parsing (OpenAI format)
    tool_calls = _try_parse_json_tool_calls(response_text, valid_tools)
    if tool_calls is not None:
        return InferenceResult(tool_calls=tool_calls, needs_retry=False)

    # Step 2: Try rescue parsing
    if valid_tools:
        rescued = rescue_tool_call(response_text, valid_tools)
        if rescued is not None:
            return InferenceResult(
                tool_calls=rescued,
                needs_retry=False,
                retry_reason="rescued",
            )

    # Step 3: Check if it's a text response (no tool calls attempted)
    stripped = response_text.strip()
    if stripped and not valid_tools:
        return InferenceResult(tool_calls=[], text=stripped, needs_retry=False)

    # Step 4: Empty or malformed — needs retry
    if not stripped:
        return InferenceResult(
            tool_calls=[],
            needs_retry=True,
            retry_reason="empty_response",
        )

    # Step 5: Contains text but tools are available — needs retry with nudge
    return InferenceResult(
        tool_calls=[],
        text=stripped,
        needs_retry=True,
        retry_reason="text_instead_of_tool_call",
    )


def _try_parse_json_tool_calls(
    text: str, valid_tools: set[str]
) -> list[ToolCall] | None:
    """Attempt to parse text as OpenAI-style JSON tool calls."""
    import json

    text = text.strip()
    if not text.startswith("{"):
        return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(data, dict):
        # {"name": "...", "arguments": {...}}
        name = data.get("name") or data.get("tool") or ""
        if name in valid_tools:
            args = data.get("arguments", data.get("args", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if isinstance(args, dict):
                return [ToolCall(tool=name, args=args)]

        # {"function": {"name": "...", "arguments": {...}}}
        if "function" in data:
            fn = data["function"]
            name = fn.get("name", "")
            if name in valid_tools:
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if isinstance(args, dict):
                    return [ToolCall(tool=name, args=args)]

        # {"tool": "...", "args": {...}}
        tool = data.get("tool") or data.get("action") or ""
        if tool in valid_tools:
            args = data.get("args") or data.get("parameters") or data.get("params") or {}
            if isinstance(args, dict):
                return [ToolCall(tool=tool, args=args)]

    return None
