"""
ResponseValidator — validates and optionally rescues LLM responses.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from hermes_forge.core.workflow import LLMResponse, TextResponse, ToolCall
from hermes_forge.guardrails.nudge import Nudge

# Default retry nudge for bare text responses
_DEFAULT_RETRY_NUDGE = (
    "Your response did not include valid tool calls. "
    "Please respond with a JSON tool call using the available tools."
)


@dataclass
class ValidationResult:
    """Result of validating an LLM response."""
    tool_calls: list[ToolCall] = None
    needs_retry: bool = False
    nudge: Nudge | None = None
    raw_response: str = ""


class ResponseValidator:
    """Validates and optionally rescues LLM responses.

    Checks:
    1. Is it a list[ToolCall] or TextResponse?
    2. If list[ToolCall]: are tool names valid? Are args valid dicts?
    3. If TextResponse: attempt rescue parsing (if enabled), else retry.
    """

    def __init__(
        self,
        tool_names: list[str],
        rescue_enabled: bool = True,
        retry_nudge_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._tool_names = set(tool_names)
        self._rescue_enabled = rescue_enabled
        self._retry_nudge_fn = retry_nudge_fn

    def validate(self, response: LLMResponse) -> ValidationResult:
        """Validate an LLM response.

        Returns a ValidationResult. If needs_retry is True, inject the
        nudge and call the LLM again.
        """
        if isinstance(response, TextResponse):
            return self._handle_text(response)

        # Must be a list[ToolCall]
        if not isinstance(response, list):
            return ValidationResult(
                needs_retry=True,
                nudge=Nudge("retry", _DEFAULT_RETRY_NUDGE),
                raw_response=str(response),
            )

        validated: list[ToolCall] = []
        for tc in response:
            if not isinstance(tc, ToolCall):
                return ValidationResult(
                    needs_retry=True,
                    nudge=Nudge("retry", _DEFAULT_RETRY_NUDGE),
                    raw_response=str(response),
                )

            # Check tool name
            if tc.tool not in self._tool_names:
                nudge_content = (
                    f"Unknown tool '{tc.tool}'. Available tools: "
                    f"{', '.join(sorted(self._tool_names))}."
                )
                return ValidationResult(
                    needs_retry=True,
                    nudge=Nudge("unknown_tool", nudge_content),
                    raw_response=str(response),
                )

            # Check args shape
            if not isinstance(tc.args, dict):
                nudge_content = (
                    f"Tool '{tc.tool}' was called with malformed arguments "
                    f"(expected a JSON object/dict). "
                    f"Please provide valid JSON arguments."
                )
                return ValidationResult(
                    needs_retry=True,
                    nudge=Nudge("malformed_args", nudge_content),
                    raw_response=str(response),
                )

            validated.append(tc)

        if not validated:
            return ValidationResult(
                needs_retry=True,
                nudge=Nudge("retry", _DEFAULT_RETRY_NUDGE),
            )

        return ValidationResult(tool_calls=validated)

    def _handle_text(self, response: TextResponse) -> ValidationResult:
        """Handle a TextResponse — attempt rescue parsing or return retry."""
        if self._rescue_enabled:
            rescued = rescue_tool_call(response.content, self._tool_names)
            if rescued:
                return ValidationResult(tool_calls=rescued)

        # Build nudge
        if self._retry_nudge_fn:
            content = self._retry_nudge_fn(response.content)
        else:
            content = _DEFAULT_RETRY_NUDGE

        return ValidationResult(
            needs_retry=True,
            nudge=Nudge("retry", content),
            raw_response=response.content,
        )


def rescue_tool_call(text: str, valid_tools: set[str]) -> list[ToolCall] | None:
    """Attempt to rescue malformed tool calls from text responses.

    Tries (in order):
    1. JSON in a fenced code block
    2. Mistral [TOOL_CALLS] format
    3. Qwen <tool_call> XML
    4. Naked JSON
    """
    # Strategy 1: JSON in code fence
    json_match = re.search(r"```(?:json)?\s*(\{.+?\}|\[.+?\])\s*```", text, re.DOTALL)
    if json_match:
        result = _parse_json_tool_call(json_match.group(1), valid_tools)
        if result:
            return result

    # Strategy 2: Mistral [TOOL_CALLS] format
    # e.g. [TOOL_CALLS] tool_name({"key": "value"})
    mistral_match = re.search(
        r"\[TOOL_CALLS\]\s*(\w+)\s*\((\{.*?\})\)", text, re.DOTALL
    )
    if mistral_match:
        tool_name = mistral_match.group(1)
        if tool_name in valid_tools:
            try:
                args = json.loads(mistral_match.group(2))
                if isinstance(args, dict):
                    return [ToolCall(tool=tool_name, args=args)]
            except json.JSONDecodeError:
                pass

    # Strategy 3: Qwen <tool_call> XML
    qwen_match = re.search(
        r"<tool_call>\s*(\w+)\s*(.*?)\s*</tool_call>", text, re.DOTALL
    )
    if qwen_match:
        tool_name = qwen_match.group(1)
        if tool_name in valid_tools:
            args_text = qwen_match.group(2).strip()
            if args_text:
                try:
                    args = json.loads(args_text)
                    if isinstance(args, dict):
                        return [ToolCall(tool=tool_name, args=args)]
                except json.JSONDecodeError:
                    pass
            return [ToolCall(tool=tool_name, args={})]

    # Strategy 4: Naked JSON
    naked_match = re.search(r"\{[^{}]+\{.*?\}|^\{.*?\}$", text, re.DOTALL)
    if naked_match:
        result = _parse_json_tool_call(naked_match.group(0), valid_tools)
        if result:
            return result

    return None


def _parse_json_tool_call(json_str: str, valid_tools: set[str]) -> list[ToolCall] | None:
    """Try to parse a JSON tool call object or array."""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if isinstance(data, dict):
        name = data.get("name") or data.get("tool") or data.get("function", {}).get("name")
        if name and name in valid_tools:
            args = data.get("arguments") or data.get("args") or data.get("parameters") or data.get("function", {}).get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            if isinstance(args, dict):
                return [ToolCall(tool=name, args=args)]
    elif isinstance(data, list):
        result: list[ToolCall] = []
        for item in data:
            if isinstance(item, dict):
                name = item.get("name") or item.get("tool") or item.get("function", {}).get("name")
                if name and name in valid_tools:
                    args = item.get("arguments") or item.get("args") or item.get("function", {}).get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    if isinstance(args, dict):
                        result.append(ToolCall(tool=name, args=args))
        return result if result else None
    return None
