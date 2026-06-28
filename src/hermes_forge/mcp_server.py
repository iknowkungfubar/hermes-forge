"""
Hermes Forge MCP Server — exposes guardrail tools as MCP resources.

This MCP server provides Hermes Agent with tools to:
1. Validate tool calls against known tool schemas
2. Rescue malformed tool calls from LLM text output
3. Enforce step ordering and prerequisites
4. Manage context budgets and compaction
"""

from __future__ import annotations

import json
import logging
from typing import Any

from hermes_forge import __version__

# Hard limits on tool input sizes to prevent abuse
_MAX_TOOL_NAME_LENGTH = 256
_MAX_ARGS_KEYS = 100
_MAX_TOOL_LIST_LENGTH = 500
_MAX_TEXT_LENGTH = 100000
_MAX_MESSAGE_COUNT = 10000
_MAX_BUDGET_TOKENS = 1048576

logger = logging.getLogger("hermes-forge.mcp")


def serve(
    host: str = "127.0.0.1",
    port: int = 9876,
    transport: str = "stdio",
) -> None:
    """Start the Forge MCP server.

    This exposes guardrail tools as MCP resources that Hermes can call.
    """
    if transport == "stdio":
        _serve_stdio()
    else:
        _serve_sse(host, port)


def _serve_stdio() -> None:
    """Serve via stdio transport (used by Hermes MCP client)."""
    try:
        from mcp.server import Server, NotificationOptions
        from mcp.server.models import InitializationOptions
        import mcp.server.stdio
        import mcp.types as types
    except ImportError as e:
        logger.error("MCP SDK not available: %s", e)
        raise

    app = Server("hermes-forge")

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="forge_validate_tool_call",
                description="Validate a tool call against known tool schemas. Checks tool name, argument shape, and returns any validation errors.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Name of the tool being called",
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Arguments for the tool call",
                        },
                        "available_tools": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of valid tool names",
                        },
                    },
                    "required": ["tool_name", "arguments", "available_tools"],
                },
            ),
            types.Tool(
                name="forge_rescue_tool_call",
                description="Attempt to extract a valid tool call from malformed LLM text output. Supports JSON in code fences, Mistral [TOOL_CALLS] format, Qwen <tool_call> XML, and naked JSON.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Raw LLM text response that may contain a malformed tool call",
                        },
                        "available_tools": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of valid tool names for validation",
                        },
                    },
                    "required": ["text", "available_tools"],
                },
            ),
            types.Tool(
                name="forge_check_step_ordering",
                description="Check if a tool call violates required step ordering or prerequisites. Returns whether the step can proceed and any pending requirements.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Name of the tool being called",
                        },
                        "completed_steps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Names of tools already called",
                        },
                        "required_steps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tools that must be called before the terminal tool",
                        },
                        "terminal_tools": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tools that end the workflow",
                        },
                    },
                    "required": [
                        "tool_name",
                        "completed_steps",
                        "required_steps",
                        "terminal_tools",
                    ],
                },
            ),
            types.Tool(
                name="forge_estimate_context_budget",
                description="Estimate the token count of a message list and check if compaction is needed based on the budget.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message_count": {
                            "type": "integer",
                            "description": "Number of messages in the conversation",
                        },
                        "estimated_tokens": {
                            "type": "integer",
                            "description": "Estimated token count (optional, will be calculated from message_count if not provided)",
                        },
                        "budget_tokens": {
                            "type": "integer",
                            "description": "Context budget in tokens",
                        },
                    },
                    "required": ["message_count", "budget_tokens"],
                },
            ),
            types.Tool(
                name="forge_config_workflow",
                description="Configure a Forge workflow with tools, required steps, and terminal tool. Returns the workflow configuration for use with other forge tools.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Workflow name"},
                        "description": {
                            "type": "string",
                            "description": "Workflow description",
                        },
                        "tools": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                            },
                            "description": "List of available tools",
                        },
                        "required_steps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tools that must be called before terminal",
                        },
                        "terminal_tool": {
                            "type": "string",
                            "description": "The tool that ends the workflow",
                        },
                    },
                    "required": [
                        "name",
                        "description",
                        "tools",
                        "required_steps",
                        "terminal_tool",
                    ],
                },
            ),
        ]

    @app.call_tool()
    async def call_tool(
        name: str, arguments: dict[str, Any] | None
    ) -> list[types.TextContent]:
        if arguments is None:
            arguments = {}

        try:
            if name == "forge_validate_tool_call":
                return [_validate_tool_call(arguments)]
            elif name == "forge_rescue_tool_call":
                return [_rescue_tool_call_safe(arguments)]
            elif name == "forge_check_step_ordering":
                return [_check_step_ordering(arguments)]
            elif name == "forge_estimate_context_budget":
                return [_estimate_context_budget_safe(arguments)]
            elif name == "forge_config_workflow":
                return [_config_workflow(arguments)]
            else:
                raise ValueError(f"Unknown tool: {name}")
        except (ValueError, TypeError) as e:
            import mcp.types as types

            return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]
        except Exception:
            logger.exception("Unexpected error in MCP tool call")
            import mcp.types as types

            return [
                types.TextContent(
                    type="text", text=json.dumps({"error": "Internal server error"})
                )
            ]

    async def run():
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="hermes-forge",
                    server_version=__version__,
                    capabilities=app.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

    import asyncio

    asyncio.run(run())


def _serve_sse(host: str, port: int) -> None:
    """Serve via SSE transport."""
    raise NotImplementedError("SSE transport coming soon. Use stdio for now.")


def _validate_tool_call(args: dict) -> Any:
    from hermes_forge.guardrails.response_validator import ResponseValidator
    from hermes_forge.core.workflow import ToolCall

    tool_name = args.get("tool_name", "")
    arguments = args.get("arguments", {})
    available_tools = args.get("available_tools", [])

    validator = ResponseValidator(tool_names=available_tools)
    tool_call = ToolCall(tool=tool_name, args=arguments)
    result = validator.validate([tool_call])

    import mcp.types as types

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


def _rescue_tool_call(args: dict) -> Any:
    from hermes_forge.guardrails.response_validator import rescue_tool_call as rescue

    text = args.get("text", "")
    available_tools = args.get("available_tools", [])
    valid_tools = set(available_tools)

    result = rescue(text, valid_tools)

    import mcp.types as types

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


def _rescue_tool_call_safe(args: dict) -> Any:
    """Wrapper with input validation."""
    text = args.get("text", "")
    available_tools = args.get("available_tools", [])

    # Validate input sizes
    if not isinstance(text, str):
        raise ValueError("'text' must be a string")
    if len(text) > _MAX_TEXT_LENGTH:
        raise ValueError(
            f"'text' exceeds maximum length of {_MAX_TEXT_LENGTH} characters"
        )
    if not isinstance(available_tools, list):
        raise ValueError("'available_tools' must be a list")
    if len(available_tools) > _MAX_TOOL_LIST_LENGTH:
        raise ValueError(
            f"'available_tools' exceeds maximum length of {_MAX_TOOL_LIST_LENGTH}"
        )
    if any(
        not isinstance(t, str) or len(t) > _MAX_TOOL_NAME_LENGTH
        for t in available_tools
    ):
        raise ValueError(
            f"Tool names must be strings under {_MAX_TOOL_NAME_LENGTH} characters"
        )

    return _rescue_tool_call(args)


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

    import mcp.types as types

    return types.TextContent(
        type="text",
        text=json.dumps(
            {
                "can_proceed": not result.needs_nudge,
                "pending_steps": pending,
                "completed_steps": enforcer.completed_steps,
                "all_required_done": enforcer.is_satisfied(),
                "nudge": result.nudge.content if result.needs_nudge else None,
            },
            indent=2,
        ),
    )


def _estimate_context_budget(args: dict) -> Any:
    message_count = args.get("message_count", 0)
    estimated_tokens = args.get("estimated_tokens", 0)
    budget_tokens = args.get("budget_tokens", 8192)

    if not estimated_tokens:
        estimated_tokens = message_count * 250  # rough estimate

    needs_compaction = estimated_tokens >= int(budget_tokens * 0.75)
    compaction_phase = 0
    if needs_compaction:
        if estimated_tokens >= int(budget_tokens * 0.90):
            compaction_phase = 3
        elif estimated_tokens >= int(budget_tokens * 0.75):
            compaction_phase = 1

    import mcp.types as types

    return types.TextContent(
        type="text",
        text=json.dumps(
            {
                "estimated_tokens": estimated_tokens,
                "budget_tokens": budget_tokens,
                "usage_pct": round(estimated_tokens / budget_tokens * 100, 1),
                "needs_compaction": needs_compaction,
                "recommended_compaction_phase": compaction_phase,
            },
            indent=2,
        ),
    )


def _estimate_context_budget_safe(args: dict) -> Any:
    """Wrapper with input validation."""
    message_count = args.get("message_count", 0)
    budget_tokens = args.get("budget_tokens", 8192)

    if not isinstance(message_count, (int, float)) or message_count < 0:
        raise ValueError("'message_count' must be a non-negative integer")
    if message_count > _MAX_MESSAGE_COUNT:
        raise ValueError(f"'message_count' exceeds maximum of {_MAX_MESSAGE_COUNT}")
    if not isinstance(budget_tokens, (int, float)) or budget_tokens < 1:
        raise ValueError("'budget_tokens' must be a positive integer")
    if budget_tokens > _MAX_BUDGET_TOKENS:
        raise ValueError(f"'budget_tokens' exceeds maximum of {_MAX_BUDGET_TOKENS}")

    return _estimate_context_budget(args)


def _config_workflow(args: dict) -> Any:
    import mcp.types as types

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
