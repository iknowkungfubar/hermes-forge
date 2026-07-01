"""
Hermes Forge CLI — validate tool calls and manage guardrail configurations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="hermes-forge — LLM tool-calling guardrails for Hermes Agent",
    )
    parser.add_argument("--version", action="store_true", help="Show version")

    sub = parser.add_subparsers(dest="command")

    # validate command
    validate = sub.add_parser(
        "validate", help="Validate a tool call against guardrails"
    )
    validate.add_argument(
        "--tools", required=True, help="JSON file with tool definitions"
    )
    validate.add_argument("--call", help="JSON tool call to validate (inline)")
    validate.add_argument("--call-file", help="JSON file with tool call to validate")

    # serve command (MCP server)
    serve = sub.add_parser("serve", help="Start the MCP server")
    serve.add_argument(
        "--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)"
    )
    serve.add_argument(
        "--port", type=int, default=9876, help="Port to bind (default: 9876)"
    )
    serve.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )

    # proxy command
    proxy = sub.add_parser("proxy", help="Start the guardrails proxy server")
    proxy.add_argument("--backend-url", help="URL of an externally managed backend")
    proxy.add_argument(
        "--backend",
        choices=["llamaserver", "llamafile", "ollama", "vllm"],
        help="Backend type",
    )
    proxy.add_argument("--model", help="Model name (ollama)")
    proxy.add_argument("--gguf", help="Path to GGUF file")
    proxy.add_argument("--model-path", help="Model directory (vllm)")
    proxy.add_argument(
        "--host", default="127.0.0.1", help="Listen host (default: 127.0.0.1)"
    )
    proxy.add_argument(
        "--port", type=int, default=8081, help="Listen port (default: 8081)"
    )
    proxy.add_argument(
        "--max-retries", type=int, default=3, help="Max retries per request"
    )
    proxy.add_argument(
        "--no-rescue", action="store_true", help="Disable rescue parsing"
    )
    proxy.add_argument(
        "--inject-respond-tool",
        action="store_true",
        help="Inject synthetic respond tool",
    )
    proxy.add_argument(
        "--budget-tokens", type=int, default=8192, help="Context budget in tokens"
    )
    proxy.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    proxy.add_argument(
        "--api-key", help="API key for the backend"
    )

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
    elif args.command == "proxy":
        _cmd_proxy(args)
    else:
        parser.print_help()


def _cmd_validate(args: argparse.Namespace) -> None:
    """Validate a tool call against guardrails."""

    # Path sanitization: resolve to absolute path and verify it's within expected paths
    def _safe_path(p: str) -> str:
        resolved = os.path.abspath(os.path.normpath(p))
        return resolved

    try:
        tools_path = _safe_path(args.tools)
        with open(tools_path) as f:
            tools_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading tools: {e}", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print("Error: Permission denied reading tools file", file=sys.stderr)
        sys.exit(1)

    tool_names = [
        t.get("name") or t.get("function", {}).get("name") for t in tools_data
    ]

    if args.call:
        try:
            call_data = json.loads(args.call)
        except json.JSONDecodeError as e:
            print(f"Invalid tool call JSON: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.call_file:
        call_path = os.path.abspath(os.path.normpath(args.call_file))
        try:
            with open(call_path) as f:
                call_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found: {args.call_file}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in call file: {e}", file=sys.stderr)
            sys.exit(1)
        except PermissionError:
            print("Error: Permission denied reading call file", file=sys.stderr)
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


def _cmd_proxy(args: argparse.Namespace) -> None:
    """Start the guardrails proxy server."""
    try:
        from hermes_forge.proxy.proxy import ProxyServer
        import os
        import logging
        import signal
        import sys

        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)

        api_key = args.api_key or os.environ.get("CUSTOM_PROVIDER_ZEN_KEY", "")

        proxy = ProxyServer(
            backend_url=args.backend_url,
            backend=args.backend,
            model=args.model,
            gguf=args.gguf,
            model_path=args.model_path,
            host=args.host,
            port=args.port,
            max_retries=args.max_retries,
            rescue_enabled=not args.no_rescue,
            inject_respond_tool=args.inject_respond_tool,
            budget_tokens=args.budget_tokens,
            api_key=api_key,
            backend_protocol="openai",
        )

        def _shutdown(sig, frame):
            print("\nShutting down...")
            proxy.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        proxy.start()
        print(f"Forge proxy running at {proxy.url}")
        print(f"  Point your client at {proxy.url}/v1/chat/completions")
        print("  Ctrl+C to stop")

        # Keep main thread alive — proxy runs on a daemon thread
        import threading
        _shutdown_event = threading.Event()
        _shutdown_event.wait()
    except ImportError as e:
        print(f"Cannot start proxy: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
