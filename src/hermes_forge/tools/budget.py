"""MCP tool handler: budget."""

from __future__ import annotations

import json
from typing import Any

import mcp.types as types


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


handle = _estimate_context_budget
