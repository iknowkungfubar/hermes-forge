"""MCP tool handler: validate."""

from __future__ import annotations

import json
from typing import Any

import mcp.types as types


def _validate_tool_call(args: dict) -> Any:
    from hermes_forge.guardrails.response_validator import ResponseValidator
    from hermes_forge.core.workflow import ToolCall

    tool_name = args.get("tool_name", "")
    arguments = args.get("arguments", {})
    available_tools = args.get("available_tools", [])

    validator = ResponseValidator(tool_names=available_tools)
    tool_call = ToolCall(tool=tool_name, args=arguments)
    result = validator.validate([tool_call])


    if result.needs_retry:
        return types.TextContent(
            type="text",
            text=json.dumps(
                {
                    "valid": False,
                    "error": result.nudge.content
                    if result.nudge
                    else "Unknown validation error",
                    "tool_name": tool_name,
                    "available_tools": available_tools,
                },
                indent=2,
            ),
        )

    return types.TextContent(
        type="text",
        text=json.dumps(
            {
                "valid": True,
                "tool_name": tool_name,
                "argument_count": len(arguments),
            },
            indent=2,
        ),
    )



handle = _validate_tool_call
