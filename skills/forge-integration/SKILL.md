---
name: hermes-forge
description: >-
  Integrate Forge LLM tool-calling guardrails into Hermes Agent — validate
  tool calls, rescue malformed output, enforce step ordering, and manage
  context budgets through the Forge MCP server and plugin.
---

# hermes-forge: LLM Tool-Calling Guardrails for Hermes

## Overview

`hermes-forge` brings Forge's reliability layer to Hermes Agent. It provides:

- **Response validation** — each tool call is checked against known schemas before execution
- **Rescue parsing** — extracts structured tool calls from malformed LLM output (code fences, Mistral format, Qwen XML)
- **Step enforcement** — requires specific tools to be called before the terminal tool
- **Context budget management** — estimates token usage and recommends compaction strategies

## Setup

### Option 1: MCP Server (Recommended)

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  hermes-forge:
    command: "python"
    args: ["-m", "hermes_forge.mcp_server_entry"]
```

Restart Hermes:

```bash
hermes mcp test hermes-forge
```

### Option 2: Plugin

Copy the plugin:

```bash
cp -r src/hermes_forge_plugin ~/.hermes/plugins/hermes-forge
hermes plugins install hermes-forge
```

## Usage

### 1. Validate a tool call

Ask Hermes:
```
Use the forge_validate_tool_call tool to validate that calling get_weather
with {"city": "London"} is valid.
```

### 2. Rescue malformed output

```
The LLM returned a tool call inside a code fence. Use forge_rescue_tool_call
to extract it from: ```json {"name": "get_weather", "arguments": {"city": "Paris"}} ```
```

### 3. Configure a workflow

```
Use forge_config_workflow to create a workflow named "weather-assistant"
with tools get_weather and send_report, required step get_weather,
terminal tool send_report.
```

### 4. Check step ordering

```
Use forge_check_step_ordering to see if I can call send_report yet.
I've completed: get_weather. Required: [get_weather]. Terminal: [send_report].
```

### 5. Monitor context budget

```
Use forge_estimate_context_budget with message_count=15 and budget_tokens=8192.
```

## Architecture

```
LLM Response → forge_validate_tool_call → Valid? → Execute
                    ↓ No
             forge_rescue_tool_call → Rescued? → Execute
                    ↓ No
             Retry with corrective nudge
```

```
Workflow → forge_config_workflow → forge_check_step_ordering → Execute
                                    ↓ Blocked
             Nudge with pending steps
```

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `forge_validate_tool_call` | Validate tool name + args against known schemas |
| `forge_rescue_tool_call` | Extract tool calls from malformed text |
| `forge_check_step_ordering` | Enforce required steps and prerequisites |
| `forge_estimate_context_budget` | Estimate token usage and compaction needs |
| `forge_config_workflow` | Configure a workflow with tools and step rules |
