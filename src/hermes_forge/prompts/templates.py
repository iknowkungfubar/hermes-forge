"""
Prompt templates — tool prompt builders and extraction helpers.
"""

from __future__ import annotations

import json
from typing import Any

from hermes_forge.core.workflow import ToolSpec


def build_tool_prompt(tool_specs: list[ToolSpec]) -> str:
    """Build a JSON tool descriptions for prompt-injected mode."""
    tools_json = []
    for spec in tool_specs:
        param_schema = spec.get_json_schema()
        tools_json.append({
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": param_schema,
            },
        })
    return json.dumps(tools_json, indent=2)


def extract_tool_call(text: str) -> dict[str, Any] | None:
    """Naive extraction of a JSON tool call from text."""
    import json
    import re

    # Try code fence first
    match = re.search(r"```(?:json)?\s*(\{.+?\}|\[.+?\])\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try naked JSON
    match = re.search(r"\{[^{}]+\{.*?\}|^\{.*?\}$", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def rescue_tool_call(text: str, valid_tools: set[str]) -> list[dict[str, Any]] | None:
    """Attempt to rescue tool calls from malformed text."""
    import re
    import json

    # Mistral format: [TOOL_CALLS] name({"key": "val"})
    mistral = re.search(r"\[TOOL_CALLS\]\s*(\w+)\s*\((\{.*?\})\)", text, re.DOTALL)
    if mistral and mistral.group(1) in valid_tools:
        try:
            args = json.loads(mistral.group(2))
            return [{"tool": mistral.group(1), "args": args}]
        except json.JSONDecodeError:
            pass

    # Qwen format: <tool_call>name\nargs</tool_call>
    qwen = re.search(r"<tool_call>\s*(\w+)\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if qwen and qwen.group(1) in valid_tools:
        args_text = qwen.group(2).strip()
        try:
            args = json.loads(args_text) if args_text else {}
            return [{"tool": qwen.group(1), "args": args}]
        except json.JSONDecodeError:
            return [{"tool": qwen.group(1), "args": {}}]

    return None
