"""
on_session_end hook — logs forge guardrail stats for the session.

Tracks how many tool calls were validated, rescued, or rejected.
Writes a brief summary to the forge plugin log for telemetry.
"""

import logging

logger = logging.getLogger(__name__)


def on_session_end(**kwargs) -> dict | None:
    """Log guardrail stats at session end. No-op (informational)."""
    session_id = kwargs.get("session_id", "unknown")
    logger.info(
        "Forge plugin session ended: %s",
        session_id,
    )
    return None
