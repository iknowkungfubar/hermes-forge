"""MCP tool handler: rescue."""

from __future__ import annotations

import json
from typing import Any

import mcp.types as types

# Maximum input text length for rescue parsing
_MAX_TEXT_LENGTH = 50000
# Maximum number of tools in the available list
_MAX_TOOL_LIST_LENGTH = 200
# Maximum tool name length
_MAX_TOOL_NAME_LENGTH = 100


def _rescue_impl(args: dict) -> Any:
    """Rescue implementation: extract tool calls from malformed LLM output."""
    from hermes_forge.guardrails.response_validator import rescue_tool_call as rescue_fn

    text = args.get("text", "")
    available_tools = args.get("available_tools", [])
    valid_tools = set(available_tools)

    # Validate input
    if not isinstance(text, str):
        raise ValueError("'text' must be a string")
    if len(text) > _MAX_TEXT_LENGTH:
        raise ValueError(f"'text' exceeds max length of {_MAX_TEXT_LENGTH}")
    if not isinstance(available_tools, list):
        raise ValueError("'available_tools' must be a list")
    if len(available_tools) > _MAX_TOOL_LIST_LENGTH:
        raise ValueError(f"'available_tools' exceeds max length of {_MAX_TOOL_LIST_LENGTH}")
    if any(not isinstance(t, str) or len(t) > _MAX_TOOL_NAME_LENGTH for t in available_tools):
        raise ValueError(f"Tool names must be strings under {_MAX_TOOL_NAME_LENGTH} characters")

    result = rescue_fn(text, valid_tools)

    return types.TextContent(
        type="text",
        text=json.dumps(
            {
                "rescued": result is not None,
                "tool_calls": [{"tool": tc.tool, "args": tc.args} for tc in result]
                if result
                else [],
            },
            indent=2,
        ),
    )


handle = _rescue_impl
