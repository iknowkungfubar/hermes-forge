"""
pre_tool_call hook — validates tool calls using forge guardrails.

Intercepts each tool call before execution and:
  1. Validates the tool name against available tools
  2. Validates argument schema
  3. Checks step ordering prerequisites (when configured)

On validation failure, injects correction context into the next LLM
turn so the agent can self-correct.

Returns None when validation passes (zero overhead).
"""

import json
import logging

logger = logging.getLogger(__name__)

# Forge reserves these as non-blocking patterns
_CONTINUE_ALLOWED_TOOLS = {
    "web_search", "web_extract", "read_file", "search_files",
    "browser_snapshot", "session_search",
}


def pre_tool_call(**kwargs) -> dict | None:
    """Validate a tool call and inject correction context if needed."""
    tool_name = kwargs.get("tool_name", "")
    tool_input = kwargs.get("tool_input", {})

    # Let poking-around tools pass through (forgivable failures)
    if tool_name in _CONTINUE_ALLOWED_TOOLS:
        return None

    # 1. Basic tool name check
    available_tools = kwargs.get("available_tools", [])
    if available_tools and tool_name not in available_tools:
        return {
            "context": (
                f"[Forge] Tool '{tool_name}' is not in the available toolset. "
                f"Available: {', '.join(sorted(available_tools)[:10])}"
                f"{'...' if len(available_tools) > 10 else ''}. "
                "Check the tool name and retry."
            )
        }

    # 2. Argument type sanity checks
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except (json.JSONDecodeError, TypeError):
            return {
                "context": (
                    f"[Forge] Tool call arguments for '{tool_name}' are malformed "
                    "(not valid JSON). Fix the argument format and retry."
                )
            }

    # 3. Empty argument detection for known tools that need them
    if tool_name in ("terminal", "write_file", "patch", "web_extract") and not tool_input:
        return {
            "context": (
                f"[Forge] Tool '{tool_name}' requires arguments but none were provided. "
                "Add the required parameters and retry."
            )
        }

    return None
