"""
vLLM client — handles model identity resolution from /v1/models.
"""

from __future__ import annotations

import logging
from typing import Any

from hermes_forge.clients.base import LLMClient, StreamChunk, TokenUsage
from hermes_forge.clients.openai_compat import OpenAICompatClient

logger = logging.getLogger("forge.client.vllm")


class VLLMClient(LLMClient):
    """Client for vLLM serving backends.

    vLLM validates the wire `model` field against its --served-model-name
    aliases (404 on mismatch). This client auto-discovers the served model
    name from /v1/models on first call.
    """

    def __init__(
        self,
        model_path: str,
        base_url: str = "http://localhost:8000/v1",
        timeout: float = 300.0,
        api_key: str = "",
    ) -> None:
        self._model_path = model_path
        self._openai = OpenAICompatClient(
            model=model_path,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        self._served_name: str | None = None

    async def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], "TokenUsage"]:
        if self._served_name is None:
            await self._discover_served_model()
        return await self._openai.send(messages, tools, **kwargs)

    async def send_stream(self, messages, tools=None, **kwargs):
        if self._served_name is None:
            await self._discover_served_model()
        async for chunk in self._openai.send_stream(messages, tools, **kwargs):
            yield chunk

    async def get_context_length(self) -> int | None:
        return await self._openai.get_context_length()

    async def aclose(self) -> None:
        await self._openai.aclose()

    async def get_served_model_name(self) -> str | None:
        """Discover the served model name from /v1/models."""
        try:
            ctx = await self._openai.get_context_length()
            # The _openai client's get_context_length queries /v1/models
            # If successful, we know the server is up
            # Re-discover with our own call
            import httpx
            base = self._openai._base_url.rstrip("/v1").rstrip("/")
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{base}/v1/models")
                if resp.status_code == 200:
                    data = resp.json()
                    models = data.get("data", [])
                    if models:
                        name = models[0].get("id")
                        if name:
                            return name
        except Exception as e:
            logger.debug("Could not discover vLLM model: %s", e)
        return None

    def _set_model_identity(self, name: str) -> None:
        """Override the model identity."""
        import dataclasses
        self._served_name = name
        # Also update the underlying OpenAI client's model field
        self._openai._model = name

    async def _discover_served_model(self) -> None:
        name = await self.get_served_model_name()
        if name:
            self._set_model_identity(name)
            logger.info("Discovered vLLM model: %s", name)
        else:
            logger.warning("Could not discover vLLM model; using provided model_path")
