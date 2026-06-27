"""
Anthropic client — supports the Anthropic Messages API.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from hermes_forge.clients.base import LLMClient, StreamChunk, ChunkType, TokenUsage

logger = logging.getLogger("forge.client.anthropic")


class AnthropicClient(LLMClient):
    """Client for Anthropic's Messages API.

    Converts between OpenAI message format and Anthropic's format.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        base_url: str = "https://api.anthropic.com/v1",
        api_key: str = "",
        timeout: float = 300.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], TokenUsage]:
        payload = self._build_payload(messages, tools, **kwargs)

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }

        try:
            resp = await self._client.post(
                f"{self._base_url}/messages",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            logger.error("Anthropic API error: %s", e)
            raise

        result, usage = self._normalize_response(data)
        return result, usage

    async def send_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        payload = self._build_payload(messages, tools, stream=True, **kwargs)
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/messages",
                    json=payload,
                    headers=headers,
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            chunk_data = line[6:].strip()
                            if chunk_data == "[DONE]":
                                yield StreamChunk(type=ChunkType.DONE)
                                return
                            try:
                                event = json.loads(chunk_data)
                            except json.JSONDecodeError:
                                continue

                            if event.get("type") == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    yield StreamChunk(
                                        type=ChunkType.TEXT,
                                        content=delta.get("text", ""),
                                    )
                                elif delta.get("type") == "input_json_delta":
                                    yield StreamChunk(
                                        type=ChunkType.TOOL_CALL,
                                        tool_args=delta.get("partial_json", ""),
                                    )
                            elif event.get("type") == "content_block_start":
                                block = event.get("content_block", {})
                                if block.get("type") == "tool_use":
                                    yield StreamChunk(
                                        type=ChunkType.TOOL_CALL,
                                        tool_name=block.get("name", ""),
                                    )
                            elif event.get("type") == "message_delta":
                                delta = event.get("delta", {})
                                if delta.get("stop_reason"):
                                    yield StreamChunk(
                                        type=ChunkType.DONE,
                                        finish_reason=delta["stop_reason"],
                                    )

            except httpx.HTTPError as e:
                logger.error("Anthropic stream error: %s", e)
                yield StreamChunk(type=ChunkType.ERROR, content=str(e))

    async def get_context_length(self) -> int | None:
        # Anthropic models have known context lengths
        ctx_map = {
            "claude-sonnet-4": 200000,
            "claude-3-5-sonnet": 200000,
            "claude-3-opus": 200000,
            "claude-3-haiku": 200000,
        }
        for key, ctx in ctx_map.items():
            if key in self._model:
                return ctx
        return 100000

    async def aclose(self) -> None:
        await self._client.aclose()

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Convert OpenAI messages to Anthropic format
        system = None
        anthro_messages = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system = content
                continue

            if role == "tool":
                # Tool results in Anthropic are "user" messages with tool_result content
                tool_call_id = msg.get("tool_call_id", "")
                anthro_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call_id,
                                "content": content,
                            }
                        ],
                    }
                )
                continue

            anthro_messages.append({"role": role, "content": content})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": anthro_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        if system:
            payload["system"] = system

        if tools:
            payload["tools"] = [
                {
                    "name": t.get("function", {}).get("name", t.get("name", "")),
                    "description": t.get("function", {}).get(
                        "description", t.get("description", "")
                    ),
                    "input_schema": t.get("function", {}).get(
                        "parameters", t.get("input_schema", {})
                    ),
                }
                for t in tools
            ]

        if stream:
            payload["stream"] = True

        return payload

    def _normalize_response(
        self, data: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], TokenUsage]:
        usage_data = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=(
                usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0)
            ),
        )

        content_blocks = data.get("content", [])
        tool_calls = []
        text_content = ""

        for block in content_blocks:
            if block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "tool": block.get("name", ""),
                        "args": block.get("input", {}),
                        "id": block.get("id", ""),
                    }
                )
            elif block.get("type") == "text":
                text_content += block.get("text", "")

        if tool_calls:
            return tool_calls, usage

        return [{"role": "assistant", "content": text_content}], usage
