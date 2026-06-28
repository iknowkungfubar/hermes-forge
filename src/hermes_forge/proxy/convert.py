"""
Message format conversion — OpenAI ↔ forge Messages.

Handles the bidirectional conversion between the OpenAI chat-completions
message format and forge's internal Message format. This is the bridge
that enables guardrails to work transparently on any OpenAI-shaped API.
"""

from __future__ import annotations

import json
from typing import Any

from hermes_forge.core.messages import (
    Message,
    MessageMeta,
    MessageRole,
    MessageType,
    ToolCallInfo,
)


def openai_to_forge(messages: list[dict[str, Any]]) -> list[Message]:
    """Convert OpenAI-format message list to forge Messages.

    This is the primary conversion used by the proxy and WorkflowRunner
    to normalize OpenAI-shaped API calls into forge's canonical format.
    """
    forge_messages: list[Message] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")
        tool_call_id = msg.get("tool_call_id")
        name = msg.get("name")

        if role == "system":
            forge_messages.append(
                Message(
                    role=MessageRole.SYSTEM,
                    content=content or "",
                    metadata=MessageMeta(type=MessageType.SYSTEM_PROMPT),
                )
            )
        elif role == "user":
            forge_messages.append(
                Message(
                    role=MessageRole.USER,
                    content=content or "",
                    metadata=MessageMeta(type=MessageType.USER_INPUT),
                )
            )
        elif role == "assistant":
            if tool_calls:
                tc_infos = []
                for i, tc in enumerate(tool_calls):
                    fn = tc.get("function", {})
                    tc_info = ToolCallInfo(
                        name=fn.get("name", ""),
                        call_id=tc.get("id", f"call_{i}"),
                        args=json.dumps(fn.get("arguments", {})),
                    )
                    tc_infos.append(tc_info)

                forge_messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=content or "",
                        metadata=MessageMeta(type=MessageType.TOOL_CALL),
                        tool_calls=tc_infos,
                    )
                )
            else:
                forge_messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=content or "",
                        metadata=MessageMeta(type=MessageType.TEXT_RESPONSE),
                    )
                )
        elif role == "tool":
            forge_messages.append(
                Message(
                    role=MessageRole.TOOL,
                    content=content or "",
                    metadata=MessageMeta(type=MessageType.TOOL_RESULT),
                    tool_name=name,
                    tool_call_id=tool_call_id,
                )
            )

    return forge_messages


def forge_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert forge Messages back to OpenAI format.

    This is used when forwarding a validated/processed message list
    to the LLM backend.
    """
    openai_messages: list[dict[str, Any]] = []
    for msg in messages:
        entry: dict[str, Any] = {"role": msg.role.value, "content": msg.content}

        if msg.tool_calls and msg.metadata.type == MessageType.TOOL_CALL:
            entry["tool_calls"] = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.args,
                    },
                }
                for tc in msg.tool_calls
            ]

        if msg.tool_call_id:
            entry["tool_call_id"] = msg.tool_call_id
            entry["name"] = msg.tool_name or ""

        openai_messages.append(entry)

    return openai_messages


def extract_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Extract tool calls from an OpenAI response message."""
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return None

    result = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args_raw = fn.get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {"raw": args_raw}
        else:
            args = args_raw
        result.append({"tool": name, "args": args, "id": tc.get("id", "")})

    return result if result else None


def build_tool_specs(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize tool definitions from various formats to OpenAI format."""
    normalized = []
    for t in tools:
        if "function" in t:
            # Already OpenAI format
            normalized.append(t)
        elif isinstance(t, dict) and all(k in t for k in ("name", "description")):
            # Forge ToolSpec-like format
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get(
                            "parameters",
                            t.get("input_schema", {"type": "object", "properties": {}}),
                        ),
                    },
                }
            )
        else:
            # Try to normalize
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.get("name", "unknown"),
                        "description": t.get("description", ""),
                        "parameters": t.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    },
                }
            )
    return normalized
