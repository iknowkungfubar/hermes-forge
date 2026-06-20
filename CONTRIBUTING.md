# Contributing to hermes-forge

## Development Setup

```bash
git clone https://github.com/iknowkungfubar/hermes-forge.git
cd hermes-forge
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/unit/test_core.py -v

# Run with coverage
python -m pytest tests/ --cov=hermes_forge --cov-report=term-missing
```

## Running Examples

```bash
python examples/01_tool_validation.py
python examples/02_step_enforcement.py
python examples/03_context_compaction.py
python examples/04_rescue_parsing.py
```

## Project Structure

```
src/hermes_forge/
  __init__.py          # Public API exports
  cli.py               # CLI entry point
  errors.py            # Error hierarchy
  mcp_server.py        # MCP server implementation
  mcp_server_entry.py  # MCP server entry point
  proxy_main.py        # Proxy server entry point
  core/
    messages.py        # Message types
    workflow.py        # ToolSpec, ToolDef, ToolCall, Workflow
    steps.py           # StepTracker
    runner.py          # WorkflowRunner
  guardrails/
    response_validator.py  # Validate + rescue parse
    step_enforcer.py       # Step order enforcement
    error_tracker.py       # Retry budget tracking
    guardrails.py          # Bundled facade
    nudge.py               # Nudge types
  clients/
    base.py            # LLMClient protocol
    openai_compat.py   # OpenAI-compatible client
    ollama.py          # Ollama client
    llamafile.py       # Llamafile client
    vllm.py            # vLLM client
    anthropic.py       # Anthropic client
  context/
    manager.py         # ContextManager
    strategies.py      # NoCompact, SlidingWindow, TieredCompact
  prompts/
    templates.py       # Tool prompt builders
    nudges.py          # Nudge templates
  tools/
    respond.py         # Synthetic respond tool
  proxy/
    convert.py         # Message format conversion
    handler.py         # Request handler
    server.py          # HTTP server
    proxy.py           # ProxyServer
src/hermes_forge_plugin/
  plugin.py            # Hermes Agent plugin
mcp_server/
  run.py               # Standalone MCP server
tests/
  unit/
    test_core.py       # Core tests (37)
    test_advanced.py   # Advanced tests (52)
    verify_rescue.py   # Functional verification
examples/
  01_tool_validation.py
  02_step_enforcement.py
  03_context_compaction.py
  04_rescue_parsing.py
skills/
  forge-integration/
    SKILL.md
```

## Coding Standards

- **Python 3.11+** — no type alias syntax (use `Union`), no `match`/`case`
- **Type hints** — all public functions must have type annotations
- **Tests** — every new feature must include tests (aim for 90%+ coverage)
- **Async** — network calls must use `asyncio` + `httpx`
- **Errors** — raise `ForgeError` subclasses, never bare `Exception`

## Pull Request Process

1. Branch from `develop`
2. Make changes with descriptive commit messages
3. Run `python -m pytest tests/ -v` — all tests must pass
4. Create PR against `main`
5. Squash-merge with descriptive message

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with new release entries
3. Tag the release: `git tag v0.x.x && git push origin v0.x.x`
4. CI automatically publishes to PyPI

## Reporting Issues

File issues at https://github.com/iknowkungfubar/hermes-forge/issues
with the appropriate label (bug, enhancement, documentation).
