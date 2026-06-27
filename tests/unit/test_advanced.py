"""
Advanced tests covering edge cases, proxy, clients, converters, and security.
"""

import pytest
from hermes_forge.core.messages import (
    Message,
    MessageMeta,
    MessageRole,
    MessageType,
)
from hermes_forge.core.workflow import (
    TextResponse,
    ToolCall,
    ToolSpec,
    ToolDef,
    Workflow,
)
from hermes_forge.guardrails.response_validator import (
    ResponseValidator,
    rescue_tool_call,
)
from hermes_forge.guardrails.step_enforcer import StepEnforcer
from hermes_forge.guardrails.guardrails import Guardrails
from hermes_forge.context.manager import ContextManager
from hermes_forge.context.strategies import (
    NoCompact,
    SlidingWindowCompact,
    TieredCompact,
)
from hermes_forge.proxy.convert import (
    openai_to_forge,
    forge_to_openai,
    extract_tool_calls,
    build_tool_specs,
)
from hermes_forge.errors import (
    ForgeError,
    ToolCallError,
    ToolExecutionError,
    StepEnforcementError,
    PrerequisiteError,
    MaxIterationsError,
    BudgetResolutionError,
    BackendError,
)


# ═══════════════════════════════════════════════════════════════
# EDGE CASE TESTS — Rescue Parsing
# ═══════════════════════════════════════════════════════════════


class TestRescueParsingEdgeCases:
    """Test rescue parsing with edge cases."""

    def test_empty_text(self):
        assert rescue_tool_call("", {"tool"}) is None

    def test_whitespace_only(self):
        assert rescue_tool_call("   \n  \t  ", {"tool"}) is None

    def test_nested_json_schema(self):
        text = '```json {"name": "complex_tool", "arguments": {"nested": {"deep": {"value": 42}}}} ```'
        calls = rescue_tool_call(text, {"complex_tool"})
        assert calls is not None
        assert calls[0].args["nested"]["deep"]["value"] == 42

    def test_array_args(self):
        text = (
            '```json {"name": "batch_process", "arguments": {"items": [1, 2, 3]}} ```'
        )
        calls = rescue_tool_call(text, {"batch_process"})
        assert calls is not None
        assert calls[0].args["items"] == [1, 2, 3]

    def test_mistral_with_reasoning_prefix(self):
        text = (
            'I\'ll look up the weather.\n[TOOL_CALLS] get_weather({"city": "London"})'
        )
        calls = rescue_tool_call(text, {"get_weather"})
        assert calls is not None
        assert calls[0].args["city"] == "London"

    def test_qwen_multiline_args(self):
        text = '<tool_call>\nquery_database\n{\n  "sql": "SELECT * FROM users",\n  "limit": 10\n}\n</tool_call>'
        calls = rescue_tool_call(text, {"query_database"})
        assert calls is not None
        assert calls[0].args["sql"] == "SELECT * FROM users"

    def test_code_fence_no_json_label(self):
        text = '```\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n```'
        calls = rescue_tool_call(text, {"get_weather"})
        assert calls is not None
        assert calls[0].args["city"] == "Paris"

    def test_malformed_json_in_fence(self):
        text = '```json {"name": "tool", "arguments": {broken} ```'
        calls = rescue_tool_call(text, {"tool"})
        # Should not crash — returns None on parse failure
        assert calls is None or len(calls) == 0

    def test_unicode_in_args(self):
        text = '```json {"name": "translate", "arguments": {"text": "こんにちは", "target": "en"}} ```'
        calls = rescue_tool_call(text, {"translate"})
        assert calls is not None
        assert calls[0].args["text"] == "こんにちは"

    def test_multiple_tools_in_json_array(self):
        text = '```json [{"name": "tool_a", "arguments": {"x": 1}}, {"name": "tool_b", "arguments": {"y": 2}}] ```'
        calls = rescue_tool_call(text, {"tool_a", "tool_b"})
        assert calls is not None
        assert len(calls) == 2

    def test_very_long_text_near_boundary(self):
        # 5000+ chars with embedded tool call
        prefix = "A" * 2500
        suffix = "B" * 2500
        text = f'{prefix}```json {{"name": "process", "arguments": {{"data": "test"}}}} ```{suffix}'
        calls = rescue_tool_call(text, {"process"})
        assert calls is not None

    def test_no_tool_match_in_array(self):
        text = '```json [{"name": "unknown_a", "arguments": {}}, {"name": "unknown_b", "arguments": {}}] ```'
        calls = rescue_tool_call(text, {"known_tool"})
        assert calls is None


# ═══════════════════════════════════════════════════════════════
# EDGE CASE TESTS — StepEnforcer
# ═══════════════════════════════════════════════════════════════


class TestStepEnforcerEdgeCases:
    def test_no_required_steps(self):
        enforcer = StepEnforcer(required_steps=[], terminal_tools=frozenset(["done"]))
        assert enforcer.is_satisfied()
        result = enforcer.check([ToolCall(tool="done", args={})])
        assert not result.needs_nudge

    def test_multiple_terminal_tools(self):
        enforcer = StepEnforcer(
            required_steps=["auth"],
            terminal_tools=frozenset(["report", "export"]),
        )
        enforcer.record("auth", {})
        assert enforcer.is_satisfied()
        result = enforcer.check([ToolCall(tool="export", args={})])
        assert not result.needs_nudge

    def test_arg_matched_prerequisite(self):
        enforcer = StepEnforcer(
            required_steps=[],
            terminal_tools=frozenset(["terminal"]),
            tool_prerequisites={
                "terminal": [{"tool": "login", "match_arg": "user"}],
            },
        )
        # Check terminal without login having been called
        enforcer.record("terminal", {"user": "admin"})
        # prereq check: login with match_arg="user" was not called
        result = enforcer.check_prerequisites(
            [ToolCall(tool="terminal", args={"user": "admin"})]
        )
        # Should be blocked because login(user=admin) was not called
        assert result.needs_nudge

    def test_duplicate_step_recording(self):
        enforcer = StepEnforcer(
            required_steps=["step_a"], terminal_tools=frozenset(["done"])
        )
        enforcer.record("step_a", {})
        enforcer.record("step_a", {})  # duplicate
        assert enforcer.is_satisfied()
        assert enforcer.completed_steps == ["step_a", "step_a"]

    def test_summary_hint_with_pending(self):
        enforcer = StepEnforcer(
            required_steps=["a", "b", "c"],
            terminal_tools=frozenset(["done"]),
        )
        hint = enforcer.summary_hint()
        assert "Required steps remaining" in hint
        assert "a" in hint
        assert "b" in hint
        assert "c" in hint

    def test_summary_hint_all_complete(self):
        enforcer = StepEnforcer(
            required_steps=["a"], terminal_tools=frozenset(["done"])
        )
        enforcer.record("a", {})
        hint = enforcer.summary_hint()
        assert "All required steps completed" in hint

    def test_prerequisite_exhaustion(self):
        """Prerequisite violations accumulate."""
        enforcer = StepEnforcer(
            required_steps=[],
            terminal_tools=frozenset(["done"]),
            tool_prerequisites={"done": ["prereq_a"]},
        )
        for _ in range(5):
            enforcer.check_prerequisites([ToolCall(tool="done", args={})])
        assert enforcer.prereq_violations >= 3


# ═══════════════════════════════════════════════════════════════
# EDGE CASE TESTS — Context Compaction
# ═══════════════════════════════════════════════════════════════


class TestCompactionEdgeCases:
    def test_tiered_phase_1_only(self):
        strategy = TieredCompact(
            keep_recent=1, compact_threshold=0.5, phase_thresholds=(0.1, 1.0, 1.0)
        )
        messages = [
            Message(role=MessageRole.SYSTEM, content="sys"),
            Message(role=MessageRole.USER, content="user"),
        ]
        for i in range(5):
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=f"msg_{i}",
                    metadata=MessageMeta(MessageType.TEXT_RESPONSE, step_index=i),
                )
            )

        result, phase = strategy.compact(messages, 50)
        assert phase >= 1, "Should compact with tight budget"

    def test_tiered_phase_3_deep(self):
        strategy = TieredCompact(keep_recent=1, compact_threshold=0.5)
        messages = [
            Message(role=MessageRole.SYSTEM, content="sys"),
            Message(role=MessageRole.USER, content="user"),
        ]
        for i in range(10):
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content="x" * 200,
                    metadata=MessageMeta(MessageType.TEXT_RESPONSE, step_index=i),
                )
            )
            messages.append(
                Message(
                    role=MessageRole.TOOL,
                    content=f"result_{i}" * 100,
                    metadata=MessageMeta(MessageType.TOOL_RESULT, step_index=i),
                    tool_name="tool",
                )
            )

        result, phase = strategy.compact(messages, 1000)
        assert phase == 3, "Should reach phase 3 aggressive compaction"

    def test_sliding_window_compact(self):
        strategy = SlidingWindowCompact(keep_recent=2, compact_threshold=0.5)
        messages = [
            Message(role=MessageRole.SYSTEM, content="system prompt"),
            Message(role=MessageRole.USER, content="user query"),
        ]
        for i in range(20):
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=f"iteration_{i}" * 50,
                    metadata=MessageMeta(MessageType.TEXT_RESPONSE, step_index=i),
                )
            )

        result, phase = strategy.compact(messages, 500)
        # Should compact: keep system + user + recent iterations
        assert 2 <= len(result) < len(messages)

    def test_no_compact_never_triggers(self):
        strategy = NoCompact()
        messages = [Message(role=MessageRole.USER, content="hi")] * 1000
        result, phase = strategy.compact(messages, 1)
        assert len(result) == 1000
        assert phase == 0

    def test_context_manager_callbacks(self):
        events = []

        def on_compact(event):
            events.append(event)

        cm = ContextManager(
            strategy=TieredCompact(keep_recent=1, compact_threshold=0.1),
            budget_tokens=50,
            on_compact=on_compact,
        )

        messages = [
            Message(role=MessageRole.SYSTEM, content="sys"),
            Message(role=MessageRole.USER, content="user"),
        ]
        for i in range(5):
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content="x" * 100,
                    metadata=MessageMeta(MessageType.TEXT_RESPONSE, step_index=i),
                )
            )

        # Should trigger compaction
        compacted = cm.compact(messages)
        assert len(compacted) < len(messages)


# ═══════════════════════════════════════════════════════════════
# EDGE CASE TESTS — Guardrails
# ═══════════════════════════════════════════════════════════════


class TestGuardrailsEdgeCases:
    def test_record_with_terminal_happy_path(self):
        guard = Guardrails(
            tool_names=["step_a", "done"],
            terminal_tool="done",
            required_steps=["step_a"],
        )
        result = guard.record([("step_a", {}), ("done", {})])
        assert result is True  # workflow complete

    def test_record_without_terminal(self):
        guard = Guardrails(
            tool_names=["step_a", "done"],
            terminal_tool="done",
            required_steps=["step_a"],
        )
        result = guard.record([("step_a", {})])
        assert result is False  # workflow not complete

    def test_fatal_on_max_retries(self):
        guard = Guardrails(
            tool_names=["tool"],
            terminal_tool="terminal",
            max_retries=2,
        )
        for _ in range(3):
            guard.check(TextResponse(content="no"))
        result = guard.check(TextResponse(content="no"))
        assert result.action == "fatal"

    def test_fatal_on_step_block_exhaustion(self):
        guard = Guardrails(
            tool_names=["step_a", "terminal"],
            terminal_tool="terminal",
            required_steps=["step_a"],
            max_premature_attempts=2,
        )
        from hermes_forge.core.workflow import ToolCall

        guard.check([ToolCall(tool="terminal", args={})])
        guard.check([ToolCall(tool="terminal", args={})])
        result = guard.check([ToolCall(tool="terminal", args={})])
        assert result.action == "fatal"

    def test_tool_error_channel(self):
        guard = Guardrails(
            tool_names=["valid_tool"],
            terminal_tool="done",
        )
        result = guard.check([ToolCall(tool="unknown_tool", args={})])
        assert result.action in ("retry", "tool_error")


# ═══════════════════════════════════════════════════════════════
# EDGE CASE TESTS — Message Conversion
# ═══════════════════════════════════════════════════════════════


class TestMessageConversion:
    def test_openai_to_forge_system(self):
        oai = [{"role": "system", "content": "You are a helpful assistant."}]
        forge = openai_to_forge(oai)
        assert len(forge) == 1
        assert forge[0].role == MessageRole.SYSTEM

    def test_openai_to_forge_tool_calls(self):
        oai = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "London"}',
                        },
                    }
                ],
            }
        ]
        forge = openai_to_forge(oai)
        assert len(forge) == 1
        assert forge[0].metadata.type == MessageType.TOOL_CALL
        assert forge[0].tool_calls is not None
        assert forge[0].tool_calls[0].name == "get_weather"

    def test_forge_to_openai_roundtrip(self):
        forge_messages = [
            Message(role=MessageRole.SYSTEM, content="sys"),
            Message(role=MessageRole.USER, content="user"),
        ]
        oai = forge_to_openai(forge_messages)
        assert len(oai) == 2
        assert oai[0]["role"] == "system"
        assert oai[1]["role"] == "user"

    def test_extract_tool_calls(self):
        msg = {
            "tool_calls": [
                {"id": "c1", "function": {"name": "tool", "arguments": '{"x": 1}'}},
            ]
        }
        result = extract_tool_calls(msg)
        assert result is not None
        assert result[0]["tool"] == "tool"

    def test_extract_tool_calls_none(self):
        msg = {"content": "hello"}
        assert extract_tool_calls(msg) is None

    def test_build_tool_specs_openai_format(self):
        tools = [
            {"type": "function", "function": {"name": "test", "description": "desc"}}
        ]
        specs = build_tool_specs(tools)
        assert len(specs) == 1
        assert specs[0]["function"]["name"] == "test"

    def test_build_tool_specs_forge_format(self):
        tools = [
            {
                "name": "forge_tool",
                "description": "A forge tool",
                "parameters": {"type": "object"},
            }
        ]
        specs = build_tool_specs(tools)
        assert len(specs) == 1


# ═══════════════════════════════════════════════════════════════
# EDGE CASE TESTS — Errors
# ═══════════════════════════════════════════════════════════════


class TestErrorTypes:
    def test_forge_error_base(self):
        e = ForgeError()
        assert isinstance(e, Exception)

    def test_tool_call_error(self):
        e = ToolCallError()
        assert "ToolCallError" in type(e).__name__

    def test_tool_execution_error(self):
        cause = ValueError("API error")
        e = ToolExecutionError(tool_name="get_weather", cause=cause)
        assert "get_weather" in str(e)
        assert "API error" in str(e)

    def test_step_enforcement_error(self):
        e = StepEnforcementError(
            terminal_tool="report",
            attempts=3,
            pending_steps=["auth", "fetch"],
        )
        assert "report" in str(e)
        assert "'auth'" in str(e) or "auth" in str(e)

    def test_prerequisite_error(self):
        e = PrerequisiteError(
            tool_name="export",
            violations=2,
            missing_prereqs=["login"],
        )
        assert "export" in str(e)
        assert "login" in str(e) or "login" in str(e.message)

    def test_max_iterations_error(self):
        e = MaxIterationsError(
            max_iterations=10,
            completed_steps=["a"],
            pending_steps=["b"],
        )
        assert "10" in str(e)

    def test_budget_resolution_error(self):
        cause = ConnectionError("refused")
        e = BudgetResolutionError(cause=cause)
        assert "refused" in str(e)

    def test_backend_error(self):
        e = BackendError(status_code=503, body="overloaded")
        assert "503" in str(e)


# ═══════════════════════════════════════════════════════════════
# EDGE CASE TESTS — Workflow Validation
# ═══════════════════════════════════════════════════════════════


class TestWorkflowEdgeCases:
    def test_terminal_tool_frozenset(self):
        from pydantic import BaseModel

        class P(BaseModel):
            pass

        def f():
            return ""

        wf = Workflow(
            name="test",
            description="test",
            tools={
                "a": ToolDef(
                    spec=ToolSpec(name="a", description="a", parameters=P), callable=f
                ),
                "b": ToolDef(
                    spec=ToolSpec(name="b", description="b", parameters=P), callable=f
                ),
            },
            required_steps=[],
            terminal_tool=["a", "b"],
            system_prompt_template="test",
        )
        assert "a" in wf.terminal_tools
        assert "b" in wf.terminal_tools

    def test_terminal_cannot_be_required(self):
        from pydantic import BaseModel

        class P(BaseModel):
            pass

        def f():
            return ""

        with pytest.raises(ValueError, match="cannot also be a required step"):
            Workflow(
                name="test",
                description="test",
                tools={
                    "a": ToolDef(
                        spec=ToolSpec(name="a", description="a", parameters=P),
                        callable=f,
                    )
                },
                required_steps=["a"],
                terminal_tool="a",
                system_prompt_template="test",
            )

    def test_missing_prerequisite_tool(self):
        from pydantic import BaseModel

        class P(BaseModel):
            pass

        def f():
            return ""

        with pytest.raises((ValueError, KeyError)):
            Workflow(
                name="test",
                description="test",
                tools={
                    "a": ToolDef(
                        spec=ToolSpec(name="a", description="a", parameters=P),
                        callable=f,
                        prerequisites=["nonexistent"],
                    )
                },
                required_steps=[],
                terminal_tool="a",
                system_prompt_template="test",
            )


# ═══════════════════════════════════════════════════════════════
# SECURITY EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════


class TestSecurityEdgeCases:
    def test_rescue_prevents_injection_via_tool_name(self):
        """Tool names with shell metacharacters should not cause issues."""
        text = '```json {"name": "tool; rm -rf /", "arguments": {}} ```'
        calls = rescue_tool_call(text, {"safe_tool"})
        assert calls is None  # Won't match valid tools

    def test_huge_arguments_rejected_safely(self):
        """Extremely large args should not crash the validator."""
        validator = ResponseValidator(tool_names=["tool"])
        huge_args = {"data": "x" * 100000}
        result = validator.validate([ToolCall(tool="tool", args=huge_args)])
        assert (
            not result.needs_retry
        )  # Should pass — size isn't validated but shouldn't crash

    def test_empty_tool_name(self):
        validator = ResponseValidator(tool_names=["valid"])
        result = validator.validate([ToolCall(tool="", args={})])
        assert result.needs_retry

    def test_nan_inf_values_in_args(self):

        validator = ResponseValidator(tool_names=["tool"])
        result = validator.validate(
            [ToolCall(tool="tool", args={"value": float("inf")})]
        )
        # Should not crash — just pass through
        assert not result.needs_retry

    def test_rescue_with_html_injection(self):
        text = '<tool_call>get_weather\n{"city": "<script>alert(1)</script>"}\n</tool_call>'
        calls = rescue_tool_call(text, {"get_weather"})
        assert calls is not None
        assert "<script>" in calls[0].args.get("city", "")
