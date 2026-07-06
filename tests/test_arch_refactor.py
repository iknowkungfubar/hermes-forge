"""Tests for architecture refactoring — tools and response modules."""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestToolHandlers:
    """Test the extracted MCP tool handlers."""

    def test_validate_tool_call(self):
        from hermes_forge.tools.validate import handle
        result = handle({
            "tool_name": "test_tool",
            "arguments": {"key": "value"},
            "available_tools": ["test_tool", "other"],
        })
        text = json.loads(result.text)
        assert text["valid"] is True
        assert text["tool_name"] == "test_tool"

    def test_validate_tool_call_invalid(self):
        from hermes_forge.tools.validate import handle
        result = handle({
            "tool_name": "unknown",
            "arguments": {},
            "available_tools": ["known_tool"],
        })
        text = json.loads(result.text)
        assert text["valid"] is False

    def test_rescue_tool_call(self):
        from hermes_forge.tools.rescue import handle
        result = handle({
            "text": 'Some text {"tool": "test", "args": {}} more text',
            "available_tools": ["test"],
        })
        text = json.loads(result.text)
        assert "rescued" in text

    def test_step_ordering(self):
        from hermes_forge.tools.step_order import handle
        result = handle({
            "tool_name": "final_step",
            "completed_steps": ["step1", "step2"],
            "required_steps": ["step1", "step2"],
            "terminal_tools": ["final_step"],
        })
        text = json.loads(result.text)
        assert text.get("can_proceed", True) is True

    def test_context_budget(self):
        from hermes_forge.tools.budget import handle
        result = handle({
            "message_count": 10,
            "budget_tokens": 10000,
        })
        text = json.loads(result.text)
        assert "estimated_tokens" in text

    def test_workflow_config(self):
        from hermes_forge.tools.workflow import handle
        result = handle({
            "name": "test_workflow",
            "description": "A test workflow",
            "tools": [{"name": "tool1", "description": "First tool"}],
            "required_steps": ["tool1"],
            "terminal_tool": "tool1",
        })
        text = json.loads(result.text)
        assert text["status"] == "configured"

    def test_mcp_server_imports(self):
        from hermes_forge.mcp_server import TOOLS, HANDLERS
        assert len(TOOLS) == 5
        assert "forge_validate_tool_call" in HANDLERS
        assert "forge_rescue_tool_call" in HANDLERS
        assert "forge_check_step_ordering" in HANDLERS
        assert "forge_estimate_context_budget" in HANDLERS
        assert "forge_config_workflow" in HANDLERS


class TestResponseModule:
    """Test the extracted proxy response module."""

    def test_build_response_envelope(self):
        from hermes_forge.proxy.response import build_response_envelope
        result = build_response_envelope(
            choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hello"}}],
            model="test-model",
        )
        assert result["model"] == "test-model"
        assert "id" in result

    def test_build_text_response(self):
        from hermes_forge.proxy.response import build_text_response
        result = build_text_response("Hello!", "test-model")
        assert result["choices"][0]["message"]["content"] == "Hello!"

    def test_build_error_response(self):
        from hermes_forge.proxy.response import build_error_response
        result = build_error_response("Something broke", "test-model")
        assert "error" in result["choices"][0]["message"]["content"]

    def test_is_tool_response(self):
        from hermes_forge.proxy.response import is_tool_response
        assert is_tool_response([{"tool_calls": [{"function": {"name": "test"}}]}]) is True
        assert is_tool_response([{"content": "text"}]) is False
        assert is_tool_response([]) is False

    def test_parse_tool_calls(self):
        from hermes_forge.proxy.response import parse_tool_calls
        result = parse_tool_calls([{"delta": {"tool_calls": [{"id": "call_1", "function": {"name": "test", "arguments": '{"key": "value"}'}}]}}])
        assert len(result) == 1
        assert result[0]["name"] == "test"


class TestPluginHook:
    """Test the refactored plugin hook."""

    def test_validate_arguments_valid(self):
        from hermes_forge_plugin.hooks.pre_tool_call import _validate_arguments
        assert _validate_arguments("test", {"key": "value"}) is None

    def test_validate_arguments_empty(self):
        from hermes_forge_plugin.hooks.pre_tool_call import _validate_arguments
        result = _validate_arguments("terminal", "")
        assert result is not None
        assert "requires arguments" in result["context"]

    def test_validate_arguments_empty_dict(self):
        from hermes_forge_plugin.hooks.pre_tool_call import _validate_arguments
        result = _validate_arguments("write_file", {})
        assert result is not None
        assert "requires arguments" in result["context"]
