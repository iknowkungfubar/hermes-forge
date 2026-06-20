"""
Nudge dataclasses — corrective messages to re-prompt the model.
"""

from __future__ import annotations

from dataclasses import dataclass

TOOL_CHANNEL_KINDS = frozenset({"step_enforcement", "prerequisite_skip"})
TOOL_ERROR_KINDS = frozenset({"unknown_tool", "malformed_args"})


@dataclass
class Nudge:
    """A corrective message to re-prompt the model."""
    kind: str  # retry, step_enforcement, prerequisite_skip, unknown_tool, malformed_args
    content: str
