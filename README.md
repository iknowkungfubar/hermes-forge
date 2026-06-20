# hermes-forge — LLM Tool-Calling Guardrails for Hermes Agent

A self-hosted MCP server and Hermes Agent plugin that brings [Forge](https://github.com/antoinezambelli/forge)'s guardrail framework to Hermes Agent. This project reverse-engineers Forge's core architecture into a reusable, extensible MCP server with connected skills and tools.

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/iknowkungfubar/hermes-forge)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## What is Forge?

[Forge](https://github.com/antoinezambelli/forge) is a reliability layer for self-hosted LLM tool-calling by Antoine Zambelli. It takes an 8B local model from single digits to 84% across 26 eval scenarios. Forge provides:

- **Response validation** — catch malformed or unknown tool calls before execution
- **Rescue parsing** — extract structured calls from 5+ malformed formats (Mistral, Qwen, code fences, naked JSON)
- **Step enforcement** — require specific tools be called before the terminal tool
- **Context management** — tiered compaction for long-running agentic loops
- **Proxy mode** — drop-in OpenAI-compatible proxy with transparent guardrails

## What is hermes-forge?

hermes-forge reimplements Forge's guardrail architecture as:

- **MCP Server** — exposes guardrail tools (validate, rescue, step-order, context-budget) as MCP resources Hermes can call
- **Hermes Plugin** — hooks into Hermes Agent's tool pipeline to apply guardrails transparently
- **Skills** — teaches Hermes how to use Forge patterns in daily workflows
- **Python Library** — the core forge logic available as a reusable `hermes_forge` package

## Installation

### Quick Install

```bash
pip install hermes-forge
```

### For MCP Server

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  hermes-forge:
    command: "python"
    args: ["-m", "hermes_forge.mcp_server_entry"]
```

Then restart Hermes and verify:

```bash
hermes mcp test hermes-forge
```

### For Plugin

```bash
cp -r src/hermes_forge_plugin ~/.hermes/plugins/hermes-forge
hermes plugins install hermes-forge
```

## Quick Start

### Validate a Tool Call

```python
from hermes_forge.guardrails.response_validator import ResponseValidator
from hermes_forge.core.workflow import ToolCall

validator = ResponseValidator(tool_names=["get_weather", "send_email"])
result = validator.validate([
    ToolCall(tool="get_weather", args={"city": "London"})
])

print("Valid:", not result.needs_retry)
```

### Rescue Malformed Output

```python
from hermes_forge.guardrails.response_validator import rescue_tool_call

text = '```json {"name": "get_weather", "arguments": {"city": "Paris"}} ```'
calls = rescue_tool_call(text, {"get_weather", "send_email"})
if calls:
    print(f"Rescued: {calls[0].tool}({calls[0].args})")
```

### Enforce Step Ordering

```python
from hermes_forge.guardrails.step_enforcer import StepEnforcer

enforcer = StepEnforcer(
    required_steps=["authenticate", "fetch_data"],
    terminal_tools=frozenset(["export_report"]),
)

# After completing steps:
enforcer.record("authenticate", {"user": "admin"})
enforcer.record("fetch_data", {"query": "sales"})

# Check if terminal can proceed:
from hermes_forge.core.workflow import ToolCall
result = enforcer.check([ToolCall(tool="export_report", args={})])
print("Can export?", not result.needs_nudge)
```

## Development

```bash
git clone https://github.com/iknowkungfubar/hermes-forge.git
cd hermes-forge
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Hermes Agent                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ MCP Client   │  │ Plugin       │  │ Skills    │ │
│  │ (discovers)  │  │ (hooks into  │  │ (prompts  │ │
│  │  tools)      │  │  pipeline)   │  │  teach)   │ │
│  └──────┬───────┘  └──────┬───────┘  └───────────┘ │
└─────────┼─────────────────┼─────────────────────────┘
          │                 │
┌─────────▼─────────────────▼─────────────────────────┐
│               hermes-forge MCP Server                │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Validate │  │ Rescue   │  │ Step Enforcer    │   │
│  │ Tool Call│  │ Parsing  │  │ + Context Budget │   │
│  └──────────┘  └──────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
src/hermes_forge/
  core/            # Messages, ToolDef, Workflow, Steps
  guardrails/      # ResponseValidator, StepEnforcer, ErrorTracker
  context/         # ContextManager, TieredCompact, SlidingWindow
  prompts/         # Tool prompt builders, nudges
  tools/           # Synthetic respond tool
  mcp_server.py    # MCP server implementation
  cli.py           # CLI entry point
src/hermes_forge_plugin/  # Hermes plugin
mcp_server/               # Standalone MCP server wrapper
skills/                   # Hermes skills
tests/                    # Test suite
```

## Related

- [Forge (original)](https://github.com/antoinezambelli/forge) — A reliability layer for self-hosted LLM tool-calling by Antoine Zambelli
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) — Open-source AI agent framework by Nous Research
- [MCP (Model Context Protocol)](https://modelcontextprotocol.io) — Protocol for AI tool calling

## License

MIT — Copyright (c) 2026 Hermes Forge Contributors. Based on Forge by Antoine Zambelli.
