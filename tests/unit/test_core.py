"""
Basic tests for hermes-forge guardrails.
"""

import json
import pytest
from hermes_forge.core.messages import (
    Message,
    MessageMeta,
    MessageRole,
    MessageType,
    ToolCallInfo,
)
from hermes_forge.core.workflow import (
    TextResponse,
    ToolCall,
    ToolSpec,
    ToolDef,
    Workflow,
)
from hermes_forge.core.steps import StepTracker
from hermes_forge.core.inference import run_inference
from hermes_forge.guardrails.response_validator import (
    ResponseValidator,
    rescue_tool_call,
)
from hermes_forge.guardrails.step_enforcer import StepEnforcer
from hermes_forge.guardrails.error_tracker import ErrorTracker
from hermes_forge.guardrails.guardrails import Guardrails
from hermes_forge.context.manager import ContextManager
from hermes_forge.context.strategies import (
    NoCompact,
    SlidingWindowCompact,
    TieredCompact,
)


class TestMessages:
    def test_message_defaults(self):
        msg = Message(role=MessageRole.USER, content="hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "hello"
        assert msg.metadata.type == MessageType.TEXT_RESPONSE
        assert msg.tool_calls is None

    def test_tool_call_info(self):
        info = ToolCallInfo(
            name="get_weather", call_id="call_1", args='{"city": "London"}'
        )
        assert info.name == "get_weather"
        assert info.call_id == "call_1"


class TestToolSpec:
    def test_from_json_schema(self):
        spec = ToolSpec.from_json_schema(
            name="get_weather",
            description="Get weather for a city",
            schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        )
        assert spec.name == "get_weather"
        assert "city" in spec.get_json_schema().get("properties", {})

    def test_from_json_schema_with_nested(self):
        spec = ToolSpec.from_json_schema(
            name="book_flight",
            description="Book a flight",
            schema={
                "type": "object",
                "properties": {
                    "destination": {"type": "string"},
                    "passenger": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                        "required": ["name"],
                    },
                },
                "required": ["destination"],
            },
        )
        assert spec.name == "book_flight"


class TestResponseValidator:
    def test_valid_tool_call(self):
        validator = ResponseValidator(tool_names=["get_weather"])
        result = validator.validate(
            [ToolCall(tool="get_weather", args={"city": "London"})]
        )
        assert not result.needs_retry
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1

    def test_unknown_tool(self):
        validator = ResponseValidator(tool_names=["get_weather"])
        result = validator.validate(
            [ToolCall(tool="send_email", args={"to": "a@b.com"})]
        )
        assert result.needs_retry
        assert result.nudge is not None
        assert "Unknown tool" in result.nudge.content

    def test_malformed_args(self):
        validator = ResponseValidator(tool_names=["get_weather"])
        result = validator.validate([ToolCall(tool="get_weather", args="not_a_dict")])
        assert result.needs_retry
        assert "malformed" in result.nudge.content.lower()

    def test_text_response_default(self):
        validator = ResponseValidator(tool_names=["get_weather"])
        result = validator.validate(TextResponse(content="I'll help you"))
        assert result.needs_retry

    def test_empty_list(self):
        validator = ResponseValidator(tool_names=["get_weather"])
        result = validator.validate([])
        assert result.needs_retry


class TestRescueParsing:
    def test_json_code_fence(self):
        text = '```json {"name": "get_weather", "arguments": {"city": "London"}} ```'
        calls = rescue_tool_call(text, {"get_weather"})
        assert calls is not None
        assert calls[0].tool == "get_weather"
        assert calls[0].args == {"city": "London"}

    def test_mistral_format(self):
        text = '[TOOL_CALLS] get_weather({"city": "Paris"})'
        calls = rescue_tool_call(text, {"get_weather"})
        assert calls is not None
        assert calls[0].tool == "get_weather"

    def test_qwen_format(self):
        text = '<tool_call>get_weather\n{"city": "Tokyo"}\n</tool_call>'
        calls = rescue_tool_call(text, {"get_weather"})
        assert calls is not None
        assert calls[0].tool == "get_weather"

    def test_no_valid_tool(self):
        text = '```json {"name": "unknown_tool", "arguments": {}} ```'
        calls = rescue_tool_call(text, {"get_weather"})
        assert calls is None


class TestStepEnforcer:
    def test_terminal_blocked_by_required_steps(self):
        enforcer = StepEnforcer(
            required_steps=["authenticate"],
            terminal_tools=frozenset(["export"]),
        )
        from hermes_forge.core.workflow import ToolCall

        result = enforcer.check([ToolCall(tool="export", args={})])
        assert result.needs_nudge

    def test_terminal_allowed_after_required(self):
        enforcer = StepEnforcer(
            required_steps=["authenticate"],
            terminal_tools=frozenset(["export"]),
        )
        enforcer.record("authenticate", {"user": "admin"})
        from hermes_forge.core.workflow import ToolCall

        result = enforcer.check([ToolCall(tool="export", args={})])
        assert not result.needs_nudge

    def test_is_satisfied(self):
        enforcer = StepEnforcer(
            required_steps=["a", "b"],
            terminal_tools=frozenset(["c"]),
        )
        assert not enforcer.is_satisfied()
        enforcer.record("a", {})
        assert not enforcer.is_satisfied()
        enforcer.record("b", {})
        assert enforcer.is_satisfied()

    def test_max_premature_attempts(self):
        enforcer = StepEnforcer(
            required_steps=["required_tool"],
            terminal_tools=frozenset(["terminal"]),
            max_premature_attempts=2,
        )
        from hermes_forge.core.workflow import ToolCall

        for _ in range(2):
            enforcer.check([ToolCall(tool="terminal", args={})])
        assert enforcer.premature_exhausted


class TestErrorTracker:
    def test_retries_exhausted(self):
        tracker = ErrorTracker(max_retries=3, max_tool_errors=2)
        for _ in range(3):
            tracker.record_retry()
        assert tracker.retries_exhausted

    def test_tool_errors_exhausted(self):
        tracker = ErrorTracker(max_retries=3, max_tool_errors=2)
        for _ in range(2):
            tracker.record_result(success=False)
        assert tracker.tool_errors_exhausted

    def test_reset_retries(self):
        tracker = ErrorTracker(max_retries=3, max_tool_errors=2)
        tracker.record_retry()
        tracker.record_retry()
        tracker.reset_retries()
        assert not tracker.retries_exhausted

    def test_success_resets_errors(self):
        tracker = ErrorTracker(max_retries=3, max_tool_errors=2)
        tracker.record_result(success=False)
        tracker.record_result(success=True)
        assert not tracker.tool_errors_exhausted


class TestGuardrails:
    def test_valid_call_execute(self):
        guard = Guardrails(
            tool_names=["get_weather"],
            terminal_tool="finish",
        )
        from hermes_forge.core.workflow import ToolCall

        result = guard.check([ToolCall(tool="get_weather", args={"city": "London"})])
        assert result.action == "execute"

    def test_unknown_tool_retry(self):
        guard = Guardrails(
            tool_names=["get_weather"],
            terminal_tool="finish",
        )
        from hermes_forge.core.workflow import ToolCall

        result = guard.check([ToolCall(tool="bad_tool", args={})])
        assert result.action in ("retry", "tool_error")

    def test_text_response_retry(self):
        guard = Guardrails(
            tool_names=["get_weather"],
            terminal_tool="finish",
        )
        result = guard.check(TextResponse(content="I'm thinking..."))
        assert result.action == "retry"

    def test_step_blocked(self):
        guard = Guardrails(
            tool_names=["step_a", "step_b", "terminal"],
            terminal_tool="terminal",
            required_steps=["step_a", "step_b"],
        )
        from hermes_forge.core.workflow import ToolCall

        result = guard.check([ToolCall(tool="terminal", args={})])
        assert result.action == "step_blocked"

    def test_fatal_after_retries(self):
        guard = Guardrails(
            tool_names=["tool"],
            terminal_tool="terminal",
            max_retries=2,
        )
        guard.check(TextResponse(content="no"))
        guard.check(TextResponse(content="no"))
        result = guard.check(TextResponse(content="no"))
        assert result.action == "fatal"


class TestStepTracker:
    def test_was_called(self):
        tracker = StepTracker()
        tracker.record("tool_a", {"x": 1})
        assert tracker.was_called("tool_a")
        assert not tracker.was_called("tool_b")

    def test_check_prerequisites(self):
        tracker = StepTracker()
        result = tracker.check_prerequisites("tool_b", {}, ["tool_a"])
        assert not result.satisfied
        assert "tool_a" in result.missing

        tracker.record("tool_a", {})
        result = tracker.check_prerequisites("tool_b", {}, ["tool_a"])
        assert result.satisfied

    def test_completed_tools(self):
        tracker = StepTracker()
        tracker.record("a", {})
        tracker.record("b", {"k": "v"})
        assert tracker.completed_tools == ["a", "b"]


class TestContextStrategies:
    def test_no_compact(self):
        strategy = NoCompact()
        messages = ["system", "user", "assistant"]
        result, phase = strategy.compact(messages, 4096)
        assert len(result) == 3
        assert phase == 0

    def test_sliding_window(self):
        strategy = SlidingWindowCompact(keep_recent=2, compact_threshold=0.5)
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="sys",
                metadata=MessageMeta(MessageType.SYSTEM_PROMPT),
            ),
            Message(
                role=MessageRole.USER,
                content="user",
                metadata=MessageMeta(MessageType.USER_INPUT),
            ),
        ]
        # Add some messages to trigger compaction
        for i in range(10):
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=f"msg_{i}",
                    metadata=MessageMeta(MessageType.TEXT_RESPONSE, step_index=i),
                )
            )

        # With small budget, compaction should trigger
        result, phase = strategy.compact(messages, 100)

        # Should still have system + user
        assert len(result) >= 2

    def test_tiered_compact_no_trigger(self):
        strategy = TieredCompact(keep_recent=2, compact_threshold=0.75)
        messages = [
            Message(role=MessageRole.SYSTEM, content="system"),
            Message(role=MessageRole.USER, content="user"),
        ]
        result, phase = strategy.compact(messages, 8192)
        assert phase == 0


class TestContextManager:
    def test_no_compact_when_below_threshold(self):
        cm = ContextManager(
            strategy=NoCompact(),
            budget_tokens=4096,
        )
        messages = [Message(role=MessageRole.USER, content="hi")]
        should, _ = cm.should_compact(messages)
        assert not should
        compacted = cm.compact(messages)
        assert len(compacted) == 1


class TestWorkflow:
    def test_valid_workflow(self):
        from pydantic import BaseModel, Field

        class Params(BaseModel):
            city: str = Field(description="City name")

        def dummy(city: str) -> str:
            return f"Weather for {city}"

        wf = Workflow(
            name="test",
            description="Test workflow",
            tools={
                "get_weather": ToolDef(
                    spec=ToolSpec(
                        name="get_weather", description="Get weather", parameters=Params
                    ),
                    callable=dummy,
                ),
            },
            required_steps=[],
            terminal_tool="get_weather",
            system_prompt_template="You are a weather assistant.",
        )
        assert wf.name == "test"
        assert "get_weather" in wf.tools

    def test_missing_tool_validation(self):
        from pydantic import BaseModel

        class Params(BaseModel):
            pass

        def dummy() -> str:
            return "ok"

        with pytest.raises(
            ValueError, match="Required step 'missing_tool' not in tools"
        ):
            Workflow(
                name="bad",
                description="Bad",
                tools={
                    "tool_a": ToolDef(
                        spec=ToolSpec(
                            name="tool_a", description="Tool A", parameters=Params
                        ),
                        callable=dummy,
                    ),
                },
                required_steps=["missing_tool"],
                terminal_tool="tool_a",
                system_prompt_template="Template",
            )

    def test_build_system_prompt(self):
        from pydantic import BaseModel

        class Params(BaseModel):
            pass

        def dummy() -> str:
            return "ok"

        wf = Workflow(
            name="test",
            description="Test",
            tools={
                "tool": ToolDef(
                    spec=ToolSpec(name="tool", description="Tool", parameters=Params),
                    callable=dummy,
                ),
            },
            required_steps=[],
            terminal_tool="tool",
            system_prompt_template="Hello {name}!",
        )
        prompt = wf.build_system_prompt(name="World")
        assert prompt == "Hello World!"


class TestRunInference:
    """Test the run_inference function and _try_parse_json_tool_calls helper."""

    def test_tool_call_openai_format(self):
        """Direct JSON tool call (OpenAI format) should return tool_calls."""
        text = json.dumps({"name": "get_weather", "arguments": {"city": "London"}})
        result = run_inference(text, tools=["get_weather"])
        assert not result.needs_retry
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool == "get_weather"
        assert result.tool_calls[0].args == {"city": "London"}

    def test_tool_call_function_wrapper_format(self):
        """Tool call with function wrapper format should work."""
        text = json.dumps(
            {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
        )
        result = run_inference(text, tools=["get_weather"])
        assert not result.needs_retry
        assert result.tool_calls[0].tool == "get_weather"
        assert result.tool_calls[0].args == {"city": "Paris"}

    def test_tool_call_action_format(self):
        """Tool call with tool/action shorthand should work."""
        text = json.dumps({"tool": "get_weather", "args": {"city": "Tokyo"}})
        result = run_inference(text, tools=["get_weather"])
        assert not result.needs_retry
        assert result.tool_calls[0].tool == "get_weather"

    def test_tool_call_string_args(self):
        """Args as a JSON string should be parsed."""
        text = json.dumps(
            {"name": "get_weather", "arguments": json.dumps({"city": "Berlin"})}
        )
        result = run_inference(text, tools=["get_weather"])
        assert not result.needs_retry
        assert result.tool_calls[0].args == {"city": "Berlin"}

    def test_rescue_parsing_code_fence(self):
        """Code fence JSON should be rescued."""
        text = '```json {"name": "get_weather", "arguments": {"city": "Rome"}} ```'
        result = run_inference(text, tools=["get_weather"])
        assert not result.needs_retry
        assert result.tool_calls[0].tool == "get_weather"

    def test_text_response_no_tools(self):
        """Pure text response with no tools should return text."""
        result = run_inference("Hello, how can I help?", tools=None)
        assert not result.needs_retry
        assert result.text == "Hello, how can I help?"

    def test_text_response_with_tools_retry(self):
        """Text response when tools are available should trigger retry."""
        result = run_inference("I'll help you with that", tools=["get_weather"])
        assert result.needs_retry
        assert result.retry_reason == "text_instead_of_tool_call"
        assert result.text == "I'll help you with that"

    def test_empty_response_retry(self):
        """Empty response should trigger retry."""
        result = run_inference("  ", tools=["get_weather"])
        assert result.needs_retry
        assert result.retry_reason == "empty_response"

    def test_unknown_tool_name_goes_to_rescue_then_retry(self):
        """Tool not in valid_tools triggers rescue then retry with text."""
        text = json.dumps({"name": "unknown_tool", "arguments": {}})
        result = run_inference(text, tools=["get_weather"])
        # JSON parsing fails (unknown tool), rescue fails (same reason),
        # falls through to text-with-tools retry
        assert result.needs_retry
        assert result.text is not None

    def test_not_json_returns_none(self):
        """Non-JSON text not starting with { should not parse."""
        from hermes_forge.core.inference import _try_parse_json_tool_calls

        assert _try_parse_json_tool_calls("hello world", {"tool"}) is None

    def test_malformed_json_returns_none(self):
        """Malformed JSON should not parse."""
        from hermes_forge.core.inference import _try_parse_json_tool_calls

        assert _try_parse_json_tool_calls("{bad json}", {"tool"}) is None

    def test_rescue_via_mistral_format(self):
        """Mistral [TOOL_CALLS] format should be rescued."""
        text = '[TOOL_CALLS] get_weather({"city": "Madrid"})'
        result = run_inference(text, tools=["get_weather"])
        assert not result.needs_retry
        assert result.tool_calls[0].tool == "get_weather"

    def test_dict_tools_parameter(self):
        """Tools as dict should be treated as valid."""
        text = json.dumps({"name": "get_weather", "arguments": {"city": "Berlin"}})
        result = run_inference(text, tools={"get_weather": None})
        assert not result.needs_retry


class TestCLI:
    def test_cli_validate_valid(self, tmp_path):
        import subprocess
        import sys

        tools_file = tmp_path / "tools.json"
        tools_file.write_text(
            json.dumps([{"name": "get_weather", "function": {"name": "get_weather"}}])
        )
        call_file = tmp_path / "call.json"
        call_file.write_text(
            json.dumps({"name": "get_weather", "arguments": {"city": "London"}})
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "hermes_forge.cli",
                "validate",
                "--tools",
                str(tools_file),
                "--call-file",
                str(call_file),
            ],
            capture_output=True,
            text=True,
        )
        print(result.stdout, result.stderr)
        assert result.returncode == 0
