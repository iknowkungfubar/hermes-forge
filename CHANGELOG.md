# Changelog

## v0.1.0 (2026-06-20)

### Initial Release

hermes-forge is a from-scratch reimplementation of the Forge LLM tool-calling
guardrail framework (https://github.com/antoinezambelli/forge) as a self-hosted
MCP server and Hermes Agent plugin.

### Core Features

- **Response Validation**: Validate tool calls against known schemas — catches
  unknown tool names and malformed arguments before execution
- **Rescue Parsing**: Extract structured tool calls from 5+ malformed formats:
  JSON code fences, Mistral `[TOOL_CALLS]`, Qwen `<tool_call>` XML, naked JSON,
  and function-call format — **9/9 formats rescued**
- **Step Enforcement**: Require specific tools to be called in order before
  the terminal tool, with optional arg-matched prerequisites
- **Error Budget Tracking**: Consecutive retry and tool-error budgets prevent
  runaway inference costs
- **Context Compaction**: Three-tiered compaction (TieredCompact, SlidingWindow,
  NoCompact) keeps long-running workflows within context budgets

### LLM Backend Clients

- `OpenAICompatClient` — any OpenAI-shaped API (llama.cpp, text-gen-webui)
- `OllamaClient` — native function calling via Ollama API
- `LlamafileClient` — native (with `--jinja`) and prompt-injected modes
- `VLLMClient` — auto-discovers served model name from `/v1/models`
- `AnthropicClient` — Anthropic Messages API with format conversion

### Proxy Server

- Drop-in OpenAI-compatible proxy that applies guardrails transparently
- External mode (`--backend-url`) and managed mode (starts backend)
- Request validation, rescue, and retry pipeline
- SSE streaming support
- Configurable retry budgets and respond-tool injection

### MCP Server

- 5 MCP tools: `forge_validate_tool_call`, `forge_rescue_tool_call`,
  `forge_check_step_ordering`, `forge_estimate_context_budget`,
  `forge_config_workflow`
- stdio transport for Hermes MCP client integration

### Hermes Plugin

- `HermesForgePlugin` with tool registry (3 registered tools)
- Lifecycle hooks for Hermes Agent integration

### Testing & Quality

- **89 unit tests** — all passing, covering edge cases and security scenarios
- CI: GitHub Actions workflow for Python 3.11/3.12
- 4 example workflows demonstrating each feature
- Input validation and error message safety

### Documentation

- README with quick start, configuration, and architecture
- AGENTS.md with SOPs for development
- SKILL.md for Hermes integration
- 4 runnable examples in `examples/`

### Limitations

- SSE transport for MCP server is not yet implemented (only stdio)
- Proxy server uses basic asyncio HTTP (no production TLS/keep-alive)
- Context budget estimation is approximate (character-count based)
