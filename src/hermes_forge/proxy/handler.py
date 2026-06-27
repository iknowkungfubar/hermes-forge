"""
Proxy HTTP request handler — bridge between HTTP API requests and Forge guardrails.

This is the core request processing pipeline:
1. Receive HTTP request (OpenAI chat-completions format)
2. Convert to forge internal Messages
3. Apply guardrails validation + rescue + step enforcement
4. Forward to LLM backend
5. Convert back to OpenAI format response
6. Return to client
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from hermes_forge.clients.base import LLMClient, TokenUsage
from hermes_forge.context.manager import ContextManager
from hermes_forge.core.workflow import ToolCall, TextResponse
from hermes_forge.guardrails.response_validator import (
    ResponseValidator,
)
from hermes_forge.proxy.convert import (
    build_tool_specs,
    forge_to_openai,
    openai_to_forge,
)
from hermes_forge.tools.respond import RESPOND_TOOL_NAME

logger = logging.getLogger("forge.proxy.handler")


class RequestHandler:
    """Handles a single proxy request through the guardrail pipeline.

    Pipeline:
    Parse → Validate → Rescue → Forward → Respond
    """

    def __init__(
        self,
        client: LLMClient,
        context_manager: ContextManager,
        max_retries: int = 3,
        max_tool_errors: int = 2,
        rescue_enabled: bool = True,
        native_passthrough: bool = True,
        inject_respond_tool: bool = False,
    ) -> None:
        self._client = client
        self._context_manager = context_manager
        self._max_retries = max_retries
        self._max_tool_errors = max_tool_errors
        self._rescue_enabled = rescue_enabled
        self._native_passthrough = native_passthrough
        self._inject_respond_tool = inject_respond_tool
        self._tool_errors = 0

    async def handle_request(self, body: dict[str, Any]) -> dict[str, Any]:
        """Process a single request through the guardrail pipeline.

        Returns an OpenAI-compatible response dict.
        """
        _start_time = time.monotonic()  # noqa: F841

        # 1. Parse
        messages = body.get("messages", [])
        tools = body.get("tools", [])
        stream = body.get("stream", False)
        model = body.get("model", "default")

        # Normalize tool definitions
        tool_specs = build_tool_specs(tools)

        # Extract tool names
        tool_names = []
        for ts in tool_specs:
            fn = ts.get("function", {})
            name = fn.get("name", ts.get("name", ""))
            if name:
                tool_names.append(name)

        # Inject synthetic respond tool if configured
        if self._inject_respond_tool and tools:
            respond_def = {
                "type": "function",
                "function": {
                    "name": RESPOND_TOOL_NAME,
                    "description": "Send a text response to the user.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The response text",
                            },
                        },
                        "required": ["content"],
                    },
                },
            }
            tool_specs.append(respond_def)
            tool_names.append(RESPOND_TOOL_NAME)

        # Set up validator
        validator = ResponseValidator(
            tool_names=tool_names,
            rescue_enabled=self._rescue_enabled,
        )

        # 2. Convert to forge messages
        forge_messages = openai_to_forge(messages)

        # 3. Compact context if needed
        if self._context_manager:
            should_compact, _ = self._context_manager.should_compact(forge_messages)
            if should_compact:
                forge_messages = self._context_manager.compact(forge_messages)

        # 4. Forward to backend
        openai_messages = forge_to_openai(forge_messages)
        backend_tools = tool_specs if self._native_passthrough else None

        retries = 0
        while retries <= self._max_retries:
            try:
                response, usage = await self._client.send(
                    openai_messages, tools=backend_tools
                )

                if self._is_tool_response(response):
                    # Apply guardrail validation
                    tool_calls = self._parse_tool_calls(response)
                    validation = validator.validate(tool_calls)

                    if validation.needs_retry:
                        retries += 1
                        logger.info(
                            "Guardrail retry %d/%d: %s",
                            retries,
                            self._max_retries,
                            validation.nudge.content
                            if validation.nudge
                            else "validation failed",
                        )
                        # Add nudge as a user message
                        if validation.nudge:
                            openai_messages.append(
                                {
                                    "role": "user"
                                    if validation.nudge.kind
                                    not in ("unknown_tool", "malformed_args")
                                    else "tool",
                                    "content": validation.nudge.content,
                                }
                            )
                        continue

                    # Build response
                    if stream:
                        return self._build_streaming_response(tool_calls, model, usage)
                    return self._build_tool_call_response(tool_calls, model, usage)

                else:
                    # Text response — check for synthetic respond
                    text_content = response[0].get("content", "") if response else ""

                    # Remove synthetic respond wrapping if present
                    if text_content:
                        cleaned = self._extract_respond_content(text_content)
                        if cleaned is not None:
                            text_content = cleaned
                    return self._build_text_response(text_content, model, usage)

            except Exception as e:
                logger.error(
                    "Backend error (attempt %d/%d): %s",
                    retries + 1,
                    self._max_retries + 1,
                    e,
                )
                retries += 1
                if retries > self._max_retries:
                    return self._build_error_response(
                        f"Backend error after {retries} attempts: {e}", model
                    )

        # Fallback
        return self._build_error_response("Max retries exceeded", model)

    def _is_tool_response(self, response: list[dict[str, Any]]) -> bool:
        """Check if the response contains tool calls."""
        if not response:
            return False
        return "tool" in response[0] or "tool_calls" in response[0]

    def _parse_tool_calls(
        self, response: list[dict[str, Any]]
    ) -> list[ToolCall] | TextResponse:
        """Parse backend response into forge ToolCall list or TextResponse."""
        tool_calls = []
        for item in response:
            if "tool" in item:
                tool_calls.append(
                    ToolCall(tool=item["tool"], args=item.get("args", {}))
                )
        if tool_calls:
            return tool_calls
        if response and "content" in response[0]:
            return TextResponse(content=response[0].get("content", ""))
        return TextResponse(content="")

    def _build_tool_call_response(
        self, tool_calls: list[ToolCall], model: str, usage: TokenUsage
    ) -> dict[str, Any]:
        """Build an OpenAI-format tool call response."""
        choices = [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tc.tool,
                                "arguments": json.dumps(tc.args),
                            },
                        }
                        for i, tc in enumerate(tool_calls)
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
        return self._build_response_envelope(choices, model, usage)

    def _build_text_response(
        self, content: str, model: str, usage: TokenUsage
    ) -> dict[str, Any]:
        choices = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ]
        return self._build_response_envelope(choices, model, usage)

    def _build_error_response(self, error: str, model: str) -> dict[str, Any]:
        return {
            "error": {
                "message": error,
                "type": "forge_error",
                "code": 500,
            }
        }

    def _build_response_envelope(
        self, choices: list[dict[str, Any]], model: str, usage: TokenUsage
    ) -> dict[str, Any]:
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        }

    def _build_streaming_response(self, tool_calls, model, usage):
        """Build a streaming response (single chunk with all tool calls)."""
        choices = []
        for i, tc in enumerate(tool_calls):
            choices.append(
                {
                    "index": i,
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "index": i,
                                "id": f"call_{i}",
                                "type": "function",
                                "function": {
                                    "name": tc.tool,
                                    "arguments": json.dumps(tc.args),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls" if i == len(tool_calls) - 1 else None,
                }
            )

        return self._build_response_envelope(choices, model, usage)

    @staticmethod
    def _extract_respond_content(content: str) -> str | None:
        """If content wraps a synthetic respond call, extract inner content."""
        import re

        # Check for respond tool call pattern in content
        match = re.search(
            r'"name"\s*:\s*"respond".*?"content"\s*:\s*"([^"]+)"',
            content,
            re.DOTALL,
        )
        if match:
            return match.group(1)
        return None
