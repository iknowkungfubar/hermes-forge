"""MCP tool handler: step order."""

from __future__ import annotations

import json
from typing import Any

import mcp.types as types


def _check_step_ordering(args: dict) -> Any:
    from hermes_forge.guardrails.step_enforcer import StepEnforcer

    tool_name = args.get("tool_name", "")
    completed = args.get("completed_steps", [])
    required = args.get("required_steps", [])
    terminal = args.get("terminal_tools", [])

    enforcer = StepEnforcer(
        required_steps=required,
        terminal_tools=frozenset(terminal),
    )

    # Replay completed steps
    for step in completed:
        enforcer.record(step, {})

    from hermes_forge.core.workflow import ToolCall

    tc = ToolCall(tool=tool_name, args={})
    result = enforcer.check([tc])

    pending = enforcer.pending()


    return types.TextContent(
        type="text",
        text=json.dumps(
            {
                "can_proceed": not result.needs_nudge,
                "pending_steps": pending,
                "completed_steps": enforcer.completed_steps,
                "all_required_done": enforcer.is_satisfied(),
                "nudge": result.nudge.content if result.needs_nudge and result.nudge is not None else None,
            },
            indent=2,
        ),
    )


handle = _check_step_ordering
