"""
Ollama client — supports native function calling.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from hermes_forge.clients.base import LLMClient, StreamChunk, ChunkType, TokenUsage

logger = logging.getLogger("forge.client.ollama")


class OllamaClient(LLMClient):
    """Client for Ollama's API with native function calling."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
        num_ctx: int = 8192,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._num_ctx = num_ctx
        self._client = httpx.AsyncClient(timeout=timeout)

    def set_num_ctx(self, num_ctx: int) -> None:
        self._num_ctx = num_ctx

    async def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], TokenUsage]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "options": {"num_ctx": self._num_ctx},
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        payload.update(kwargs)

        try:
            resp = await self._client.post(
                f"{self._base_url}/api/chat", json=payload
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            logger.error("Ollama API error: %s", e)
            raise

        usage = TokenUsage(
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=(data.get("prompt_eval_count", 0) + data.get("eval_count", 0)),
        )

        if data.get("tool_calls"):
            result = []
            for tc in data["tool_calls"]:
                fn = tc.get("function", {})
                result.append({
                    "tool": fn.get("name", ""),
                    "args": fn.get("arguments", {}),
                })
            return result, usage

        content = data.get("message", {}).get("content", "")
        reasoning = self._extract_reasoning(content)
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
            "options": {"num_ctx": self._num_ctx},
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        payload.update(kwargs)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                async with client.stream(
                    "POST", f"{self._base_url}/api/chat", json=payload
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if chunk.get("done"):
                            yield StreamChunk(type=ChunkType.DONE, finish_reason=chunk.get("done_reason"))
                            return

                        if chunk.get("tool_calls"):
                            for tc in chunk["tool_calls"]:
                                fn = tc.get("function", {})
                                yield StreamChunk(
                                    type=ChunkType.TOOL_CALL,
                                    tool_name=fn.get("name"),
                                    tool_args=json.dumps(fn.get("arguments", {})),
                                )

                        msg = chunk.get("message", {})
                        if msg.get("content"):
                            yield StreamChunk(type=ChunkType.TEXT, content=msg["content"])
                        if msg.get("reasoning"):
                            yield StreamChunk(type=ChunkType.REASONING, content=msg["reasoning"])

            except httpx.HTTPError as e:
                logger.error("Ollama stream error: %s", e)
                yield StreamChunk(type=ChunkType.ERROR, content=str(e))

    async def get_context_length(self) -> int | None:
        return self._num_ctx

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _extract_reasoning(content: str) -> str | None:
        import re
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
