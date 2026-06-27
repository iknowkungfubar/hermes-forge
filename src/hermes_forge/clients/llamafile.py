"""
Llamafile / llama.cpp client — supports native FC (with --jinja) and prompt-injected mode.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

from hermes_forge.clients.base import LLMClient, StreamChunk, TokenUsage
from hermes_forge.clients.openai_compat import OpenAICompatClient

logger = logging.getLogger("forge.client.llamafile")


class LlamafileClient(LLMClient):
    """Client for Llamafile and llama.cpp servers.

    Two modes:
    - native: forwards tools as OpenAI tool_calls (backend must have --jinja)
    - prompt: builds tool descriptions into the system prompt, parses JSON calls back
    """

    def __init__(
        self,
        gguf_path: str | Path,
        base_url: str = "http://localhost:8080/v1",
        mode: Literal["native", "prompt"] = "native",
        timeout: float = 300.0,
        api_key: str = "",
    ) -> None:
        self._gguf_path = str(gguf_path)
        self._mode = mode
        self._openai = OpenAICompatClient(
            model=self._gguf_path,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    @property
    def mode(self) -> str:
        return self._mode

    async def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], TokenUsage]:
        if self._mode == "prompt" and tools:
            # Inject tool descriptions into the system prompt
            messages = self._inject_tools_into_prompt(messages, tools)
            tools = None

        return await self._openai.send(messages, tools, **kwargs)

    async def send_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        if self._mode == "prompt" and tools:
            messages = self._inject_tools_into_prompt(messages, tools)
            tools = None

        async for chunk in self._openai.send_stream(messages, tools, **kwargs):
            yield chunk

    async def get_context_length(self) -> int | None:
        return await self._openai.get_context_length()

    async def aclose(self) -> None:
        await self._openai.aclose()

    def _inject_tools_into_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Inject tool descriptions into the system prompt for non-FC backends."""
        tools_json = json.dumps(tools, indent=2)
        tool_prompt = (
            "\n\nYou have access to these tools. When you want to call a tool, "
            "respond with a JSON object in the format:\n"
            '{"name": "tool_name", "arguments": {"arg1": "val1"}}\n\n'
            f"Tools available:\n{tools_json}"
        )

        if messages and messages[0].get("role") == "system":
            messages = list(messages)
            messages[0] = {
                "role": "system",
                "content": messages[0]["content"] + tool_prompt,
            }
        else:
            messages = [{"role": "system", "content": tool_prompt.strip()}] + list(messages)

        return messages
