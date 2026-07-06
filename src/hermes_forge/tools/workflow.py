"""MCP tool handler: workflow."""

from __future__ import annotations

import json
from typing import Any

import mcp.types as types


def _config_workflow(args: dict) -> Any:

    return types.TextContent(
        type="text",
        text=json.dumps(
            {
                "workflow_name": args.get("name", ""),
                "description": args.get("description", ""),
                "tool_count": len(args.get("tools", [])),
                "required_steps": args.get("required_steps", []),
                "terminal_tool": args.get("terminal_tool", ""),
                "status": "configured",
            },
            indent=2,
        ),
    )


handle = _config_workflow
