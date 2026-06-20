"""
Forge Proxy — programmatic API and CLI entry point.

The proxy sits between any OpenAI-compatible client and an LLM backend,
applying Forge guardrails transparently.

Two modes:
- External: User manages the backend, proxy connects to it.
- Managed: Forge starts and manages the backend.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal

from hermes_forge.clients.base import LLMClient
from hermes_forge.clients.anthropic import AnthropicClient
from hermes_forge.clients.ollama import OllamaClient
from hermes_forge.clients.openai_compat import OpenAICompatClient
from hermes_forge.clients.vllm import VLLMClient
from hermes_forge.context.manager import ContextManager
from hermes_forge.context.strategies import TieredCompact
from hermes_forge.proxy.handler import RequestHandler
from hermes_forge.proxy.server import HTTPServer

logger = logging.getLogger("forge.proxy")


class ProxyServer:
    """OpenAI-compatible proxy that applies forge guardrails transparently.

    Usage:
        # External mode
        proxy = ProxyServer(backend_url="http://localhost:8080")
        proxy.start()

        # Managed mode with Ollama
        proxy = ProxyServer(backend="ollama", model="ministral-3:8b")
        proxy.start()
    """

    def __init__(
        self,
        # External mode
        backend_url: str | None = None,
        # Managed mode
        backend: str | None = None,
        model: str | None = None,
        gguf: str | Path | None = None,
        model_path: str | Path | None = None,
        # Shared
        host: str = "127.0.0.1",
        port: int = 8081,
        max_retries: int = 3,
        max_tool_errors: int = 2,
        rescue_enabled: bool = True,
        native_passthrough: bool = True,
        inject_respond_tool: bool = False,
        budget_tokens: int = 8192,
        api_key: str = "",
        backend_protocol: Literal["openai", "anthropic"] = "openai",
        backend_port: int = 8080,
        timeout: float = 300.0,
    ) -> None:
        if backend_url is None and backend is None:
            raise ValueError("Provide either backend_url (external) or backend (managed)")

        self._host = host
        self._port = port
        self._max_retries = max_retries
        self._max_tool_errors = max_tool_errors
        self._rescue_enabled = rescue_enabled
        self._native_passthrough = native_passthrough
        self._inject_respond_tool = inject_respond_tool
        self._budget_tokens = budget_tokens
        self._timeout = timeout

        self._client = self._build_client(
            backend_url=backend_url,
            backend=backend,
            model=model,
            gguf=gguf,
            model_path=model_path,
            api_key=api_key,
            backend_protocol=backend_protocol,
            backend_port=backend_port,
        )

        self._http_server: HTTPServer | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = False

    def _build_client(
        self,
        backend_url: str | None,
        backend: str | None,
        model: str | None,
        gguf: str | Path | None,
        model_path: str | Path | None,
        api_key: str,
        backend_protocol: str,
        backend_port: int,
    ) -> tuple[LLMClient, ContextManager]:
        """Build the appropriate client for the backend."""
        ctx_manager = ContextManager(
            strategy=TieredCompact(),
            budget_tokens=self._budget_tokens,
        )

        if backend_protocol == "anthropic":
            client: LLMClient = AnthropicClient(
                model=model or "claude-sonnet-4-20250514",
                base_url=(backend_url or "https://api.anthropic.com/v1").rstrip("/"),
                api_key=api_key,
                timeout=self._timeout,
            )
            return client, ctx_manager

        if backend_url:
            # External mode
            base = backend_url.rstrip("/")
            if not base.endswith("/v1"):
                base += "/v1"

            if backend == "vllm":
                client = VLLMClient(
                    model_path=model or "default",
                    base_url=base,
                    api_key=api_key,
                    timeout=self._timeout,
                )
            else:
                client = OpenAICompatClient(
                    model=model or "default",
                    base_url=base,
                    api_key=api_key,
                    timeout=self._timeout,
                )

            return client, ctx_manager

        # Managed mode
        if backend == "ollama":
            client = OllamaClient(
                model=model or "default",
                timeout=self._timeout,
            )
        elif backend in ("llamaserver", "llamafile"):
            client = OpenAICompatClient(
                model=str(gguf or "default"),
                base_url=f"http://localhost:{backend_port}/v1",
                api_key=api_key,
                timeout=self._timeout,
            )
        elif backend == "vllm":
            client = VLLMClient(
                model_path=str(model_path or "default"),
                base_url=f"http://localhost:{backend_port}/v1",
                api_key=api_key,
                timeout=self._timeout,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return client, ctx_manager

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def start(self) -> None:
        if self._started:
            return

        ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop, args=(ready,), daemon=True,
        )
        self._thread.start()
        ready.wait(timeout=30)

        if not self._started:
            raise RuntimeError("Proxy failed to start")

        logger.info("Proxy ready at %s", self.url)

    def stop(self) -> None:
        if not self._started or self._loop is None:
            return

        asyncio.run_coroutine_threadsafe(self._async_stop(), self._loop).result(timeout=10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._started = False
        logger.info("Proxy stopped")

    def _run_loop(self, ready: threading.Event) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_start(ready))
            self._loop.run_forever()
        finally:
            self._loop.close()

    async def _async_start(self, ready: threading.Event) -> None:
        client, ctx_manager = self._client
        handler = RequestHandler(
            client=client,
            context_manager=ctx_manager,
            max_retries=self._max_retries,
            max_tool_errors=self._max_tool_errors,
            rescue_enabled=self._rescue_enabled,
            native_passthrough=self._native_passthrough,
            inject_respond_tool=self._inject_respond_tool,
        )
        self._http_server = HTTPServer(
            handler=handler,
            host=self._host,
            port=self._port,
        )
        await self._http_server.start()
        self._started = True
        ready.set()

    async def _async_stop(self) -> None:
        if self._http_server:
            await self._http_server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Forge proxy — OpenAI-compatible proxy with guardrails")

    # Mode
    parser.add_argument("--backend-url", help="URL of an externally managed backend")
    parser.add_argument("--backend", choices=["llamaserver", "llamafile", "ollama", "vllm"],
                        help="Backend type for managed mode")

    # Managed mode options
    parser.add_argument("--model", help="Model name (ollama)")
    parser.add_argument("--gguf", help="Path to GGUF file (llamaserver/llamafile)")
    parser.add_argument("--model-path", help="Model directory (vllm)")

    # Proxy options
    parser.add_argument("--host", default="127.0.0.1", help="Listen host")
    parser.add_argument("--port", type=int, default=8081, help="Listen port")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries")
    parser.add_argument("--max-tool-errors", type=int, default=2, help="Max tool errors")
    parser.add_argument("--no-rescue", action="store_true", help="Disable rescue parsing")
    parser.add_argument("--inject-respond-tool", action="store_true",
                        help="Inject synthetic respond tool")
    parser.add_argument("--budget-tokens", type=int, default=8192, help="Context budget")
    parser.add_argument("--api-key", help="API key for the backend")
    parser.add_argument("--backend-protocol", choices=["openai", "anthropic"], default="openai",
                        help="Backend wire protocol")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    proxy = ProxyServer(
        backend_url=args.backend_url,
        backend=args.backend,
        model=args.model,
        gguf=args.gguf,
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        max_retries=args.max_retries,
        max_tool_errors=args.max_tool_errors,
        rescue_enabled=not args.no_rescue,
        inject_respond_tool=args.inject_respond_tool,
        budget_tokens=args.budget_tokens,
        api_key=args.api_key or "",
        backend_protocol=args.backend_protocol,
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

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        _shutdown(0, None)


if __name__ == "__main__":
    main()
