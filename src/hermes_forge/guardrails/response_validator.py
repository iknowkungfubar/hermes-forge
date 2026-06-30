"""
ResponseValidator — validates and optionally rescues LLM responses.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from hermes_forge.core.workflow import LLMResponse, TextResponse, ToolCall
from hermes_forge.guardrails.nudge import Nudge

_DEFAULT_RETRY_NUDGE = (
    "Your response did not include valid tool calls. "
    "Please respond with a JSON tool call using the available tools."
)


@dataclass
class ValidationResult:
    """Result of validating an LLM response."""

    tool_calls: list[ToolCall] | None = None
    needs_retry: bool = False
    nudge: Nudge | None = None
    raw_response: str = ""


class ResponseValidator:
    """Validates and optionally rescues LLM responses.

    Checks:
    1. Is it list[ToolCall] or TextResponse?
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
        if isinstance(response, TextResponse):
            return self._handle_text(response)

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
        if self._rescue_enabled:
            rescued = rescue_tool_call(response.content, self._tool_names)
            if rescued:
                return ValidationResult(tool_calls=rescued)

        if self._retry_nudge_fn:
            content = self._retry_nudge_fn(response.content)
        else:
            content = _DEFAULT_RETRY_NUDGE

        return ValidationResult(
            needs_retry=True,
            nudge=Nudge("retry", content),
            raw_response=response.content,
        )


def _extract_outermost_json(text: str) -> str | None:
    """Extract the outermost balanced JSON object or array from text.

    Uses a stack-based approach to handle nested braces.
    """
    start = -1
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
        elif ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]

    return None


def rescue_tool_call(text: str, valid_tools: set[str]) -> list[ToolCall] | None:
    """Attempt to rescue malformed tool calls from text responses.

    Tries (in order):
    1. JSON in a fenced code block
    2. Mistral [TOOL_CALLS] format
    3. Qwen <tool_call> XML
    4. Naked JSON (balanced braces)
    """
    # Strategy 1: JSON in code fence
    json_match = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if json_match:
        result = _parse_json_tool_call(json_match.group(1), valid_tools)
        if result:
            return result

    # Strategy 2: Mistral [TOOL_CALLS] format
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
    # Format A (flat): <tool_call>tool_name\n{"args"}\n</tool_call>
    # Format B (nested): <tool_call>\n<tool_name>name</tool_name>\n<arguments>{"args"}</arguments>\n</tool_call>
    qwen_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if qwen_match:
        inner = qwen_match.group(1).strip()

        # Try Format B (nested): <tool_name>...</tool_name> + <arguments>...</arguments>
        name_match = re.search(r"<tool_name>\s*(.*?)\s*</tool_name>", inner, re.DOTALL)
        args_match = re.search(r"<arguments>\s*(.*?)\s*</arguments>", inner, re.DOTALL)

        if name_match:
            tool_name = name_match.group(1).strip()
            if tool_name in valid_tools:
                args = {}
                if args_match:
                    args_text = args_match.group(1).strip()
                    if args_text:
                        try:
                            parsed = json.loads(args_text)
                            if isinstance(parsed, dict):
                                args = parsed
                        except json.JSONDecodeError:
                            pass
                return [ToolCall(tool=tool_name, args=args)]

        # Try Format A (flat): first word = tool, rest = JSON
        flat_match = re.match(r"^(\w+)\s*(.*)", inner, re.DOTALL)
        if flat_match:
            tool_name = flat_match.group(1)
            if tool_name in valid_tools:
                args_text = flat_match.group(2).strip()
                if args_text:
                    try:
                        args = json.loads(args_text)
                        if isinstance(args, dict):
                            return [ToolCall(tool=tool_name, args=args)]
                    except json.JSONDecodeError:
                        pass
                return [ToolCall(tool=tool_name, args={})]

    # Strategy 4: Naked JSON with balanced braces
    outermost = _extract_outermost_json(text)
    if outermost:
        result = _parse_json_tool_call(outermost, valid_tools)
        if result:
            return result

    return None


def _parse_json_tool_call(
    json_str: str, valid_tools: set[str]
) -> list[ToolCall] | None:
    """Try to parse a JSON tool call object or array."""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if isinstance(data, dict):
        # OpenAI-style: {"name": "...", "arguments": {...}}
        name = data.get("name") or data.get("tool") or ""
        if not name and "function" in data:
            # OpenAI function-call format: {"function": {"name": "...", "arguments": {...}}}
            name = data["function"].get("name", "")
            if name in valid_tools:
                args = data["function"].get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                if isinstance(args, dict):
                    return [ToolCall(tool=name, args=args)]

        if name and name in valid_tools:
            args = (
                data.get("arguments")
                or data.get("args")
                or data.get("parameters")
                or {}
            )
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
                name = (
                    item.get("name")
                    or item.get("tool")
                    or item.get("function", {}).get("name", "")
                )
                if name and name in valid_tools:
                    args = (
                        item.get("arguments")
                        or item.get("args")
                        or item.get("function", {}).get("arguments", {})
                    )
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    if isinstance(args, dict):
                        result.append(ToolCall(tool=name, args=args))
        return result if result else None

    return None
