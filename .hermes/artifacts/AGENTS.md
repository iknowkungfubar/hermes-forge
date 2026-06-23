# AGENTS.md — hermes-forge Standard Operating Procedures

## Project Overview

This is a reverse-engineered reimplementation of Forge
(https://github.com/antoinezambelli/forge) as an MCP server and Hermes plugin.
Forge is a reliability layer for self-hosted LLM tool-calling.

## Key Files

```
src/hermes_forge/
  core/messages.py       — Message types for the agentic loop
  core/workflow.py       — ToolSpec, ToolDef, ToolCall, Workflow definition
  core/runner.py         — WorkflowRunner (agentic loop)
  core/steps.py          — StepTracker (completed steps tracking)
  guardrails/             — ResponseValidator, StepEnforcer, ErrorTracker
  context/               — ContextManager, TieredCompact/SlidingWindow/NoCompact
  prompts/               — Tool prompt builders, nudge templates
  tools/respond.py       — Synthetic respond tool
  mcp_server.py          — MCP server (stdio transport)
  cli.py                 — CLI with validate and serve subcommands
src/hermes_forge_plugin/ — Hermes Agent plugin
mcp_server/run.py        — Standalone MCP server entry point
skills/forge-integration/ — SKILL.md for Hermes
tests/unit/test_core.py  — Unit tests
```

## Development Workflow

1. **Branch from develop** — never commit directly to main
2. **Tests before code** — run `python -m pytest tests/ -v` before pushing
3. **Incremental commits** — one logical change per commit
4. **PRs to main** — squash-merge with descriptive messages
5. **Version bumps** — follow semver in pyproject.toml

## Running Tests

```bash
python -m pytest tests/ -v --tb=short
python -m pytest tests/ --cov=hermes_forge --cov-report=term-missing
```

## MCP Server

```bash
# Test stdio mode
python -m hermes_forge.mcp_server_entry

# Or via hermes config:
# mcp_servers:
#   hermes-forge:
#     command: "python"
#     args: ["-m", "hermes_forge.mcp_server_entry"]
```

## Quick Verification

```python
from hermes_forge import (
    ResponseValidator, ToolCall, Guardrails,
    TextResponse, rescue_tool_call,
)

# Validate a tool call
v = ResponseValidator(tool_names=["get_weather"])
result = v.validate([ToolCall(tool="get_weather", args={"city": "London"})])
assert not result.needs_retry

# Rescue malformed output
calls = rescue_tool_call(
    '```json {"name": "get_weather", "arguments": {"city": "Paris"}} ```',
    {"get_weather"},
)
assert calls and calls[0].args == {"city": "Paris"}
```
