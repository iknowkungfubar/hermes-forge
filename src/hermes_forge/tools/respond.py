"""
Synthetic respond tool — injected to keep models in tool-calling mode.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from hermes_forge.core.workflow import ToolSpec

RESPOND_TOOL_NAME = "respond"


class RespondParams(BaseModel):
    """Parameters for the respond tool."""
    content: str = Field(description="The text response to return to the user.")


def respond_spec() -> ToolSpec:
    """Return the ToolSpec for the synthetic respond tool."""
    return ToolSpec(
        name=RESPOND_TOOL_NAME,
        description="Send a text response to the user. Use this when you want to respond with text rather than calling a domain tool.",
        parameters=RespondParams,
    )


def respond_tool(content: str) -> str:
    """The respond tool callable — returns content as-is."""
    return content
