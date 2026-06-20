"""
OpenAI-compatible client for llama.cpp / generic OpenAI endpoints.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from hermes_forge.clients.base import LLMClient, StreamChunk, ChunkType, TokenUsage

logger = logging.getLogger("forge.client.openai")


class OpenAICompatClient(LLMClient):
    """Client for any OpenAI-compatible API endpoint.

    Works with llama.cpp, text-gen-webui, and any server that
    speaks the OpenAI chat-completions schema.
    """

    def __init__(
        self,
        model: str = "default",
        base_url: str = "http://localhost:8080/v1",
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
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
        payload.update(kwargs)

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            resp = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            logger.error("OpenAI API error: %s", e)
            raise

        choice = data["choices"][0]
        message = choice.get("message", {})
        usage_data = data.get("usage", {})

        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        if message.get("tool_calls"):
            return [self._normalize_tool_call(tc) for tc in message["tool_calls"]], usage

        # Text response
        content = message.get("content", "")
        reasoning = message.get("reasoning", None) or self._extract_reasoning(content)
        return [{"role": "assistant", "content": content, "reasoning": reasoning}], usage

    async def send_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        payload.update(kwargs)

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                async with client.stream(
                    "POST", f"{self._base_url}/chat/completions",
                    json=payload, headers=headers,
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        chunk_data = line[6:].strip()
                        if chunk_data == "[DONE]":
                            yield StreamChunk(type=ChunkType.DONE)
                            return
                        try:
                            chunk = json.loads(chunk_data)
                        except json.JSONDecodeError:
                            continue

                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        finish = chunk.get("choices", [{}])[0].get("finish_reason")

                        if delta.get("tool_calls"):
                            for tc_delta in delta["tool_calls"]:
                                fn = tc_delta.get("function", {})
                                yield StreamChunk(
                                    type=ChunkType.TOOL_CALL,
                                    tool_name=fn.get("name"),
                                    tool_args=fn.get("arguments"),
                                    finish_reason=finish,
                                )
                        elif delta.get("content"):
                            yield StreamChunk(
                                type=ChunkType.TEXT,
                                content=delta["content"],
                                finish_reason=finish,
                            )
                        elif delta.get("reasoning"):
                            yield StreamChunk(
                                type=ChunkType.REASONING,
                                content=delta["reasoning"],
                            )

                        if finish:
                            yield StreamChunk(type=ChunkType.DONE, finish_reason=finish)

            except httpx.HTTPError as e:
                logger.error("Stream error: %s", e)
                yield StreamChunk(type=ChunkType.ERROR, content=str(e))

    async def get_context_length(self) -> int | None:
        try:
            resp = await self._client.get(f"{self._base_url}/models")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", [])
            if models:
                return models[0].get("max_model_len", models[0].get("max_context_length"))
        except Exception as e:
            logger.debug("Could not get context length: %s", e)
        return None

    async def aclose(self) -> None:
        await self._client.aclose()

    def _normalize_tool_call(self, tc: dict[str, Any]) -> dict[str, Any]:
        """Normalize an OpenAI tool call to forge's format."""
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args_raw = fn.get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {"raw": args_raw}
        else:
            args = args_raw
        return {
            "tool": name,
            "args": args,
            "id": tc.get("id", ""),
        }

    @staticmethod
    def _extract_reasoning(content: str) -> str | None:
        """Extract <think> tags from content."""
        import re
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
