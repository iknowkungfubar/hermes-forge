"""
Hermes Forge CLI — validate tool calls and manage guardrail configurations.
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="hermes-forge — LLM tool-calling guardrails for Hermes Agent",
    )
    parser.add_argument("--version", action="store_true", help="Show version")

    sub = parser.add_subparsers(dest="command")

    # validate command
    validate = sub.add_parser("validate", help="Validate a tool call against guardrails")
    validate.add_argument("--tools", required=True, help="JSON file with tool definitions")
    validate.add_argument("--call", help="JSON tool call to validate (inline)")
    validate.add_argument("--call-file", help="JSON file with tool call to validate")

    # serve command (MCP server)
    serve = sub.add_parser("serve", help="Start the MCP server")
    serve.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=9876, help="Port to bind (default: 9876)")
    serve.add_argument("--transport", choices=["stdio", "sse"], default="stdio", help="MCP transport (default: stdio)")

    args = parser.parse_args()

    if args.version:
        try:
            from hermes_forge import __version__
            print(f"hermes-forge v{__version__}")
        except ImportError:
            print("hermes-forge v0.1.0")
        return

    if args.command == "validate":
        _cmd_validate(args)
    elif args.command == "serve":
        _cmd_serve(args)
    else:
        parser.print_help()


def _cmd_validate(args: argparse.Namespace) -> None:
    """Validate a tool call against guardrails."""
    try:
        with open(args.tools) as f:
            tools_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading tools: {e}", file=sys.stderr)
        sys.exit(1)

    tool_names = [t.get("name") or t.get("function", {}).get("name") for t in tools_data]

    if args.call:
        try:
            call_data = json.loads(args.call)
        except json.JSONDecodeError as e:
            print(f"Invalid tool call JSON: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.call_file:
        try:
            with open(args.call_file) as f:
                call_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading tool call: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Provide --call or --call-file", file=sys.stderr)
        sys.exit(1)

    name = call_data.get("name") or call_data.get("tool") or ""
    if name not in tool_names:
        print(f"FAIL: Unknown tool '{name}'. Available: {', '.join(tool_names)}")
        sys.exit(1)

    args_data = call_data.get("arguments") or call_data.get("args") or {}
    if not isinstance(args_data, dict):
        print(f"FAIL: Arguments for '{name}' must be a JSON object (dict)")
        sys.exit(1)

    print(f"PASS: Tool '{name}' with {len(args_data)} arg(s) is valid")


def _cmd_serve(args: argparse.Namespace) -> None:
    """Start the MCP server."""
    try:
        from hermes_forge.mcp_server import serve
        serve(host=args.host, port=args.port, transport=args.transport)
    except ImportError as e:
        print(f"Cannot start MCP server: {e}", file=sys.stderr)
        print("Install mcp package: pip install mcp", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
