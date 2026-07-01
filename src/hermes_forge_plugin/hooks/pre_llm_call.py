"""
pre_llm_call hook — injects forge guardrail instructions into every LLM call.

On the first turn, injects a compact set of tool-calling best practices:
  - Tool call formatting (JSON in code fences)
  - Rescue instructions (what to do when a previous tool call failed)
  - Step ordering awareness
  - Common pitfalls to avoid

On subsequent turns, injects a lightweight reminder when a tool call
was recently rescued or failed validation.

Returns None when no injection is needed (zero overhead).
"""

import json
import logging

logger = logging.getLogger(__name__)


def pre_llm_call(**kwargs) -> dict | None:
    """Inject forge guardrail guidance into the LLM call context."""
    is_first_turn = kwargs.get("is_first_turn", False)
    context_parts = []

    if is_first_turn:
        context_parts.append(_forge_system_guide())

    # Check if previous turn had a rescued/failed tool call
    forge_ctx = kwargs.get("forge_context") or {}
    if forge_ctx.get("last_call_rescued"):
        context_parts.append(
            "[Forge] The previous tool call was malformed and was rescued. "
            "Verify the corrected call served its purpose before continuing."
        )
    if forge_ctx.get("last_call_validated"):
        context_parts.append(
            "[Forge] The previous tool call failed validation. "
            "Re-check the required parameters and try again."
        )

    if not context_parts:
        return None
    return {"context": "\n\n".join(context_parts)}


def _forge_system_guide() -> str:
    """Return the forge tool-calling guide injected on first turn."""
    return (
        "[Forge Tool-Calling Guide]\n"
        "• Always output tool calls as a JSON object in a code fence:\n"
        '  ```json\n'
        '  {"name": "tool_name", "arguments": {"key": "value"}}\n'
        "  ```\n"
        "• If you need to call multiple tools, do them sequentially.\n"
        "• If a tool call fails, analyze the error and retry with corrected arguments.\n"
        "• Never repeat the exact same failing call — change the approach.\n"
        "• Use the minimum tool needed for the task.\n"
        "• If you're unsure about arguments, ask the user rather than guessing."
    )
