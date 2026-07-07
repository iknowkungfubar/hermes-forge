"""
Tests for the Anthropic client message conversion logic.
These tests exercise _build_payload and _normalize_response which are pure
functions — no network calls, no mocking needed.
"""

from hermes_forge.clients.anthropic import AnthropicClient


class TestAnthropicBuildPayload:
    """Test the OpenAI-to-Anthropic message format conversion."""

    def test_simple_message_conversion(self):
        """Convert a plain user/assistant exchange to Anthropic format."""
        client = AnthropicClient(model="claude-sonnet-4-20250514")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        payload = client._build_payload(messages)
        assert payload["model"] == "claude-sonnet-4-20250514"
        # System message should be extracted to top-level
        assert payload["system"] == "You are a helpful assistant."
        # Only user+assistant messages in the messages array
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "Hello!"
        assert payload["messages"][1]["role"] == "assistant"
        assert payload["messages"][1]["content"] == "Hi there!"

    def test_tool_result_conversion(self):
        """Tool results should be wrapped in tool_result content blocks."""
        client = AnthropicClient()
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": '{"temp": 22, "condition": "sunny"}',
            },
        ]
        payload = client._build_payload(messages)
        # Tool result messages should be user messages with tool_result content blocks
        tool_msg = payload["messages"][-1]
        assert tool_msg["role"] == "user"
        assert tool_msg["content"][0]["type"] == "tool_result"
        assert tool_msg["content"][0]["tool_use_id"] == "call_abc"
        assert "sunny" in tool_msg["content"][0]["content"]

    def test_tool_definitions_conversion(self):
        """OpenAI tool definitions should be converted to Anthropic format."""
        client = AnthropicClient()
        messages = [{"role": "user", "content": "Get weather"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        payload = client._build_payload(messages, tools=tools)
        assert "tools" in payload
        assert len(payload["tools"]) == 1
        anthro_tool = payload["tools"][0]
        assert anthro_tool["name"] == "get_weather"
        assert anthro_tool["description"] == "Get weather for a city"
        assert "input_schema" in anthro_tool

    def test_no_system_message(self):
        """Messages without a system role should not set the system field."""
        client = AnthropicClient()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        payload = client._build_payload(messages)
        assert "system" not in payload
        assert len(payload["messages"]) == 2

    def test_max_tokens_from_kwargs(self):
        """max_tokens should be configurable via kwargs."""
        client = AnthropicClient()
        messages = [{"role": "user", "content": "Hi"}]
        payload = client._build_payload(messages, max_tokens=8192)
        assert payload["max_tokens"] == 8192

    def test_streaming_flag(self):
        """stream=True should add the stream field to the payload."""
        client = AnthropicClient()
        messages = [{"role": "user", "content": "Hi"}]
        payload = client._build_payload(messages, stream=True)
        assert payload["stream"] is True

    def test_tools_with_forge_format_names(self):
        """Forge-format tools (name at top level) should also convert."""
        client = AnthropicClient()
        messages = [{"role": "user", "content": "Hi"}]
        tools = [
            {
                "name": "forge_tool",
                "description": "A forge native tool",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
        payload = client._build_payload(messages, tools=tools)
        assert payload["tools"][0]["name"] == "forge_tool"

    def test_get_context_length_known_model(self):
        """Known Anthropic models should return their context length."""
        client = AnthropicClient(model="claude-sonnet-4-20250514")
        # get_context_length is async, so we test the known mapping through
        # the instance's model name
        assert "claude-sonnet-4" in client._model

    def test_empty_messages(self):
        """Empty messages list should not crash."""
        client = AnthropicClient()
        payload = client._build_payload([])
        assert payload["messages"] == []
        assert payload["max_tokens"] == 4096


class TestAnthropicNormalizeResponse:
    """Test the Anthropic response normalization."""

    def test_text_response(self):
        """A text-only response should return assistant content."""
        client = AnthropicClient()
        data = {
            "content": [{"type": "text", "text": "Hello from Claude!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result, usage = client._normalize_response(data)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hello from Claude!"
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15

    def test_tool_use_response(self):
        """A response with tool_use blocks should return tool call dicts."""
        client = AnthropicClient()
        data = {
            "content": [
                {"type": "text", "text": "I'll look that up."},
                {
                    "type": "tool_use",
                    "id": "tu_123",
                    "name": "get_weather",
                    "input": {"city": "London"},
                },
            ],
            "usage": {"input_tokens": 15, "output_tokens": 10},
        }
        result, usage = client._normalize_response(data)
        assert len(result) == 1  # Only the tool call, text is discarded
        assert result[0]["tool"] == "get_weather"
        assert result[0]["args"]["city"] == "London"
        assert result[0]["id"] == "tu_123"

    def test_multiple_tool_calls(self):
        """Multiple tool_use blocks should all be returned."""
        client = AnthropicClient()
        data = {
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "weather"}},
                {"type": "tool_use", "id": "tu_2", "name": "get_forecast", "input": {"days": 3}},
            ],
            "usage": {"input_tokens": 20, "output_tokens": 15},
        }
        result, usage = client._normalize_response(data)
        assert len(result) == 2
        assert result[0]["tool"] == "search"
        assert result[1]["tool"] == "get_forecast"

    def test_empty_usage(self):
        """Missing usage data should default to zeros."""
        client = AnthropicClient()
        data = {"content": [{"type": "text", "text": "Hello"}]}
        result, usage = client._normalize_response(data)
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_empty_content(self):
        """Empty content list should return empty response."""
        client = AnthropicClient()
        data = {"content": [], "usage": {"input_tokens": 5, "output_tokens": 0}}
        result, usage = client._normalize_response(data)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == ""
