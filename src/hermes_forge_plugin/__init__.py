"""
Hermes Forge Plugin — transparent guardrails for every LLM call.

Registers hooks that inject forge tool-calling guidance into prompts
and validate tool calls before execution. No skills, no tools — just
hooks that make the agent more reliable.

To install:
  cp -r src/hermes_forge_plugin ~/.hermes/plugins/hermes-forge
  # or: hermes plugins install path/to/hermes_forge_plugin
"""

import logging

logger = logging.getLogger(__name__)

__all__: list[str] = []
