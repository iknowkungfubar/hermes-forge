"""MCP server — STDIO and SSE transport for Forge guardrail tools.

Routes incoming MCP tool calls to modular handlers under tools/.
This is the thin dispatcher after the architecture refactor —
tool logic lives in tools/{validate,rescue,step_order,budget,workflow}.py.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import mcp.types as types
from mcp.server.lowlevel import Server

from hermes_forge.tools import validate, rescue, step_order, budget, workflow

logger = logging.getLogger("forge.mcp")

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS: list[types.Tool] = [
    types.Tool(
        name="forge_validate_tool_call",
        description="Validate a tool call against known tool schemas",
        inputSchema={
            "type": "object",
            "properties": {
                "tool_name": {"type": "string"},
                "arguments": {"type": "object"},
                "available_tools": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["tool_name", "arguments", "available_tools"],
        },
    ),
    types.Tool(
        name="forge_rescue_tool_call",
        description="Extract valid tool call from malformed LLM output",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "available_tools": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["text", "available_tools"],
        },
    ),
    types.Tool(
        name="forge_check_step_ordering",
        description="Verify prerequisite steps before proceeding",
        inputSchema={
            "type": "object",
            "properties": {
                "tool_name": {"type": "string"},
                "completed_steps": {"type": "array", "items": {"type": "string"}},
                "required_steps": {"type": "array", "items": {"type": "string"}},
                "terminal_tools": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["tool_name", "completed_steps", "required_steps", "terminal_tools"],
        },
    ),
    types.Tool(
        name="forge_estimate_context_budget",
        description="Check if context budget is approaching limits",
        inputSchema={
            "type": "object",
            "properties": {
                "message_count": {"type": "integer"},
                "estimated_tokens": {"type": "integer"},
                "budget_tokens": {"type": "integer"},
            },
            "required": ["message_count", "budget_tokens"],
        },
    ),
    types.Tool(
        name="forge_config_workflow",
        description="Configure a recurring workflow with tool set and prerequisites",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "tools": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
                "required_steps": {"type": "array", "items": {"type": "string"}},
                "terminal_tool": {"type": "string"},
            },
            "required": ["name", "description", "tools", "required_steps", "terminal_tool"],
        },
    ),
]

# Dispatch map
HANDLERS: dict[str, Any] = {
    "forge_validate_tool_call": validate.handle,
    "forge_rescue_tool_call": rescue.handle,
    "forge_check_step_ordering": step_order.handle,
    "forge_estimate_context_budget": budget.handle,
    "forge_config_workflow": workflow.handle,
}


# ---------------------------------------------------------------------------
# Call tool handler
# ---------------------------------------------------------------------------


async def handle_call(name: str, arguments: dict) -> list[types.TextContent]:
    """Dispatch a tool call to the appropriate handler."""
    handler = HANDLERS.get(name)
    if handler is None:
        return [types.TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]
    try:
        result = handler(arguments)
        if isinstance(result, list):
            return result
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.exception("Tool %s failed", name)
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


# ---------------------------------------------------------------------------
# STDIO transport
# ---------------------------------------------------------------------------


def serve_stdio() -> None:
    """Serve via STDIO transport."""
    app = Server("hermes-forge")

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return TOOLS

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        return await handle_call(name, arguments)

    from mcp.server.stdio import stdio_server
    import anyio

    async def _run():
        async with stdio_server() as (read, write):
            await app.run(read, write, app.create_initialization_options())

    anyio.run(_run)


# ---------------------------------------------------------------------------
# SSE transport
# ---------------------------------------------------------------------------


def serve_sse(host: str = "127.0.0.1", port: int = 8089) -> None:
    """Serve via SSE transport."""
    app = Server("hermes-forge")

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return TOOLS

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        return await handle_call(name, arguments)

    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route
    import uvicorn

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await app.run(streams[0], streams[1], app.create_initialization_options())

    starlette_app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ]
    )

    uvicorn.run(starlette_app, host=host, port=port)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def serve(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8089) -> None:
    """Start the MCP server with the specified transport."""
    if transport == "stdio":
        serve_stdio()
    elif transport == "sse":
        serve_sse(host, port)
    else:
        raise ValueError(f"Unknown transport: {transport}")
