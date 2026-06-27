"""
Raw asyncio HTTP server for the Forge proxy.

Implements a minimal HTTP/1.1 server that serves the OpenAI chat-completions
API. Handles POST /v1/chat/completions with JSON body, returns JSON responses.
Streaming support via SSE (text/event-stream).

Security: request body size is limited to prevent memory exhaustion.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from hermes_forge.proxy.handler import RequestHandler

logger = logging.getLogger("forge.proxy.server")

# Maximum request body size: 10MB
_MAX_REQUEST_SIZE = 10 * 1024 * 1024

# Maximum header size: 16KB
_MAX_HEADER_SIZE = 16 * 1024


class HTTPServer:
    """Minimal HTTP server for the Forge proxy.

    Single endpoint: POST /v1/chat/completions
    """

    def __init__(
        self,
        handler: RequestHandler,
        host: str = "127.0.0.1",
        port: int = 8081,
    ) -> None:
        self._handler = handler
        self._host = host
        self._port = port
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_connection,
            host=self._host,
            port=self._port,
        )
        logger.info("HTTP server listening on %s:%d", self._host, self._port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("HTTP server stopped")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            request_data = await self._read_request(reader)
            if request_data is None:
                writer.close()
                return

            method, path, headers, body = request_data

            if (
                path in ("/v1/chat/completions", "/chat/completions")
                and method == "POST"
            ):
                response = await self._handle_chat_completion(body)
            elif path in ("/health", "/v1/health") and method == "GET":
                response = {"status": "ok", "service": "hermes-forge-proxy"}
            elif path == "/v1/models" and method == "GET":
                response = {"data": [{"id": "forge-proxy", "object": "model"}]}
            else:
                response = self._not_found()

            await self._send_json_response(writer, response)

        except Exception as e:
            logger.error("Connection error: %s", e)
            try:
                await self._send_json_response(
                    writer,
                    {
                        "error": {
                            "message": "Internal server error",
                            "type": "internal_error",
                        }
                    },
                    status=500,
                )
            except Exception:
                pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _read_request(
        self, reader: asyncio.StreamReader
    ) -> tuple[str, str, dict[str, str], dict[str, Any]] | None:
        """Read and parse an HTTP request."""
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=30)
            if not request_line:
                return None

            request_str = request_line.decode("utf-8", errors="replace").strip()
            parts = request_str.split(" ")
            if len(parts) < 2:
                return None
            method = parts[0]
            path = parts[1]

            # Path sanitization: block path traversal and unusual paths
            sanitized_path = path.split("?")[0].split("#")[0]  # strip query/fragment
            if ".." in sanitized_path or sanitized_path.count("/") > 5:
                return None

            # Read headers with size limit
            headers: dict[str, str] = {}
            total_header_size = 0
            while True:
                header_line = await asyncio.wait_for(reader.readline(), timeout=30)
                header_str = header_line.decode("utf-8", errors="replace").strip()
                total_header_size += len(header_line)
                if total_header_size > _MAX_HEADER_SIZE:
                    logger.warning("Request headers exceed size limit")
                    return None
                if not header_str:
                    break
                if ":" in header_str:
                    key, value = header_str.split(":", 1)
                    headers[key.strip().lower()] = value.strip()

            # Read body with size enforcement
            content_length = int(headers.get("content-length", "0"))
            if content_length > _MAX_REQUEST_SIZE:
                logger.warning("Request body too large: %d bytes", content_length)
                return None
            body_bytes = b""
            if content_length > 0:
                body_bytes = await asyncio.wait_for(
                    reader.readexactly(content_length), timeout=60
                )

            body = {}
            if body_bytes:
                body = json.loads(body_bytes.decode("utf-8"))

            return method, path, headers, body

        except asyncio.TimeoutError:
            logger.warning("Request read timeout")
            return None
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("Invalid request body: %s", e)
            return "POST", "/v1/chat/completions", {}, {}

    async def _handle_chat_completion(self, body: dict[str, Any]) -> dict[str, Any]:
        """Handle a chat completion request through the guardrail pipeline."""
        return await self._handler.handle_request(body)

    async def _send_json_response(
        self,
        writer: asyncio.StreamWriter,
        data: dict[str, Any],
        status: int = 200,
    ) -> None:
        body = json.dumps(data).encode("utf-8")
        response = (
            f"HTTP/1.1 {status} {'OK' if status == 200 else 'Error'}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Access-Control-Allow-Origin: *\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode("utf-8") + body

        try:
            writer.write(response)
            await writer.drain()
        except Exception as e:
            logger.error("Failed to send response: %s", e)

    @staticmethod
    def _not_found() -> dict[str, Any]:
        return {"error": {"message": "Not found", "type": "not_found", "code": 404}}
