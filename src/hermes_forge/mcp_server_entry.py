"""
Hermes Forge — MCP Server entry point.

This can be run directly or configured in Hermes config.yaml:

```yaml
mcp_servers:
  hermes-forge:
    command: "python"
    args: ["-m", "hermes_forge.mcp_server_entry"]
```

The server exposes tools that Hermes can call to validate tool calls,
rescue malformed output, enforce step ordering, and manage context budgets.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Entry point for the MCP server."""
    from hermes_forge.mcp_server import serve

    # Default: stdio transport for Hermes MCP client integration
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    host = "127.0.0.1"
    port = 9876

    if transport == "sse":
        if len(sys.argv) > 2:
            host = sys.argv[2]
        if len(sys.argv) > 3:
            port = int(sys.argv[3])

    serve(host=host, port=port, transport=transport)


if __name__ == "__main__":
    main()
