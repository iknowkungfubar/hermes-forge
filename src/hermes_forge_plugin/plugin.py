"""
Hermes Forge Plugin — integrates forge guardrails into the Hermes tool pipeline.

This plugin hooks into Hermes Agent's tool-calling pipeline to apply Forge's
guardrails transparently: response validation, rescue parsing, step enforcement,
and context management.

To install in Hermes:
    hermes plugins install hermes-forge
    # or copy to ~/.hermes/plugins/hermes-forge/
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("hermes-forge.plugin")


class HermesForgePlugin:
    """Hermes Agent plugin that integrates Forge guardrail capabilities."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._tool_registry: dict[str, Any] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the plugin. Called by Hermes on startup."""
        if self._initialized:
            return

        logger.info("Initializing hermes-forge plugin v0.1.0")
        self._initialized = True

    def get_tool_registry(self) -> dict[str, Any]:
        """Return the tool registry for Hermes tool discovery."""
        if not self._initialized:
            self.initialize()

        return {
            "forge_validate_tool_call": {
                "description": "Validate a tool call against known tool schemas",
                "handler": self._handle_validate,
                "schema": {
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string"},
                        "arguments": {"type": "object"},
                        "available_tools": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["tool_name", "arguments", "available_tools"],
                },
            },
            "forge_rescue_tool_call": {
                "description": "Rescue malformed tool calls from LLM text output",
                "handler": self._handle_rescue,
                "schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "available_tools": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["text", "available_tools"],
                },
            },
            "forge_config_workflow": {
                "description": "Configure a Forge workflow with tools, required steps, and terminal tool",
                "handler": self._handle_config_workflow,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "tools": {"type": "array"},
                        "required_steps": {"type": "array", "items": {"type": "string"}},
                        "terminal_tool": {"type": "string"},
                    },
                    "required": ["name", "description", "tools", "required_steps", "terminal_tool"],
                },
            },
        }

    def _handle_validate(self, params: dict[str, Any]) -> str:
        """Validate a tool call."""
        import json
        from hermes_forge.guardrails.response_validator import ResponseValidator
        from hermes_forge.core.workflow import ToolCall

        tool_name = params.get("tool_name", "")
        arguments = params.get("arguments", {})
        available = params.get("available_tools", [])

        validator = ResponseValidator(tool_names=available)
        result = validator.validate([ToolCall(tool=tool_name, args=arguments)])

        return json.dumps({
            "valid": not result.needs_retry,
            "error": result.nudge.content if result.needs_retry else None,
            "tool_name": tool_name,
        })

    def _handle_rescue(self, params: dict[str, Any]) -> str:
        """Rescue malformed tool calls from text."""
        import json
        from hermes_forge.guardrails.response_validator import rescue_tool_call

        text = params.get("text", "")
        available = set(params.get("available_tools", []))
        result = rescue_tool_call(text, available)

        return json.dumps({
            "rescued": result is not None,
            "tool_calls": [
                {"tool": tc.tool, "args": tc.args}
                for tc in result
            ] if result else [],
        })

    def _handle_config_workflow(self, params: dict[str, Any]) -> str:
        """Configure a workflow."""
        import json

        workflow_id = f"{params.get('name', 'workflow')}-{id(params)}"
        self._tool_registry[workflow_id] = {
            "name": params.get("name"),
            "description": params.get("description"),
            "tools": params.get("tools", []),
            "required_steps": params.get("required_steps", []),
            "terminal_tool": params.get("terminal_tool"),
        }

        return json.dumps({
            "status": "configured",
            "workflow_id": workflow_id,
            "tool_count": len(params.get("tools", [])),
        })


# Plugin entry point — Hermes discovers this
plugin = HermesForgePlugin
