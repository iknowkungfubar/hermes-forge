"""Response formatting utilities for the proxy handler.

Extracted from RequestHandler to create testable seams for
response envelope building, tool call formatting, and error responses.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any


def build_response_envelope(
    choices: list[dict[str, Any]],
    model: str,
    usage: dict[str, int] | Any | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    """Build an OpenAI-compatible chat completion response envelope.

    The usage parameter accepts both dict and dataclass TokenUsage objects.
    """
    body: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk" if stream else "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }
    if usage:
        if isinstance(usage, dict):
            body["usage"] = usage
        else:
            # Handle TokenUsage dataclass
            body["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
    return body


def build_tool_call_response(
    tool_name: str,
    tool_args: dict[str, Any],
    tool_call_id: str,
    model: str,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a response containing a single tool call."""
    choices = [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "delta" if True else "message": {  # stream vs non-stream handled by caller
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args),
                        },
                    }
                ],
            },
        }
    ]
    return build_response_envelope(choices, model, usage)


def build_text_response(
    text: str,
    model: str,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a response containing text content."""
    choices = [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": text,
            },
        }
    ]
    return build_response_envelope(choices, model, usage)


def build_error_response(error: str, model: str) -> dict[str, Any]:
    """Build an error response envelope."""
    choices = [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps({"error": error}),
            },
        }
    ]
    return build_response_envelope(choices, model)


def is_tool_response(response: list[dict[str, Any]]) -> bool:
    """Check if the LLM response contains tool calls."""
    if not response:
        return False
    first = response[0]
    return bool(first.get("tool_calls")) or bool(
        first.get("delta", {}).get("tool_calls")
    )


def parse_tool_calls(response: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract tool calls from an LLM response."""
    tool_calls = []
    for choice in response:
        delta = choice.get("delta", choice)
        for tc in delta.get("tool_calls", []):
            tool_calls.append({
                "id": tc.get("id", ""),
                "name": tc.get("function", {}).get("name", ""),
                "arguments": tc.get("function", {}).get("arguments", "{}"),
            })
    return tool_calls
