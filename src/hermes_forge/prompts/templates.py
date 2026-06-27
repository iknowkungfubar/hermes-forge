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
        tools_json.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": param_schema,
                },
            }
        )
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
