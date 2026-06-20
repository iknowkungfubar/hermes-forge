"""
Nudge templates — standard retry and step-enforcement nudges.
"""

from __future__ import annotations


def retry_nudge(raw_response: str) -> str:
    """Default retry nudge for malformed responses."""
    return (
        "Your response didn't include valid tool calls. "
        "Please respond with the correct tool call format."
    )


def step_nudge(tool_name: str, pending: list[str]) -> str:
    """Nudge for premature terminal tool attempts."""
    return (
        f"Cannot call '{tool_name}' yet. "
        f"Complete these steps first: {', '.join(pending)}."
    )
