"""
Standalone MCP server process for Hermes Forge.

This file is a thin wrapper that the `mcp_server/` directory points to.
Run directly:
    python mcp_server/run.py

Or configure in Hermes config.yaml:
```yaml
mcp_servers:
  hermes-forge:
    command: "python"
    args: ["-m", "mcp_server.run"]
```
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hermes_forge.mcp_server import serve

if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    serve(transport=transport)
