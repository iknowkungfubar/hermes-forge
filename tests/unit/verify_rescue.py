# ruff: noqa: E402
"""Quick rescue parsing verification."""

from hermes_forge.guardrails.response_validator import rescue_tool_call

# Test 1: JSON with triple backticks
text = '```json {"name": "get_weather", "arguments": {"city": "Paris"}} ```'
calls = rescue_tool_call(text, {"get_weather"})
assert calls, f"JSON code fence rescue failed for: {text}"
assert calls[0].args == {"city": "Paris"}, f"Expected Paris, got {calls[0].args}"
print(f"PASS: JSON code fence → {calls[0].tool}({calls[0].args})")

# Test 2: Mistral
text2 = '[TOOL_CALLS] get_weather({"city": "London"})'
calls2 = rescue_tool_call(text2, {"get_weather"})
assert calls2, f"Mistral rescue failed for: {text2}"
print(f"PASS: Mistral → {calls2[0].tool}({calls2[0].args})")

# Test 3: Qwen
text3 = '<tool_call>get_weather\n{"city": "Tokyo"}\n</tool_call>'
calls3 = rescue_tool_call(text3, {"get_weather"})
assert calls3, f"Qwen rescue failed for: {text3}"
print(f"PASS: Qwen → {calls3[0].tool}({calls3[0].args})")

# Test 4: Unknown
text4 = '```json {"name": "unknown_tool", "arguments": {}} ```'
calls4 = rescue_tool_call(text4, {"get_weather"})
assert calls4 is None, f"Expected None for unknown tool, got {calls4}"
print("PASS: Unknown tool returns None")

# Test 5: All validations
from hermes_forge.guardrails.response_validator import ResponseValidator  # noqa: E402
from hermes_forge.core.workflow import ToolCall  # noqa: E402

v = ResponseValidator(tool_names=["get_weather", "send_email"])
r = v.validate([ToolCall(tool="get_weather", args={"city": "London"})])
assert not r.needs_retry, "Valid call should not need retry"
print("PASS: Valid tool call passes")

r = v.validate([ToolCall(tool="unknown", args={"x": 1})])
assert r.needs_retry, "Unknown tool should need retry"
print("PASS: Unknown tool rejected")

# Test 6: Guardrails facade
from hermes_forge.guardrails.guardrails import Guardrails

g = Guardrails(
    tool_names=["a", "b", "finish"], terminal_tool="finish", required_steps=["a", "b"]
)
r = g.check([ToolCall(tool="finish", args={})])
assert r.action == "step_blocked", f"Expected step_blocked, got {r.action}"
g.record([("a", {}), ("b", {})])
r = g.check([ToolCall(tool="finish", args={})])
assert r.action == "execute", f"Expected execute, got {r.action}"
print("PASS: Guardrails step enforcement")

# Test 7: StepTracker prerequisites
from hermes_forge.core.steps import StepTracker

t = StepTracker()
t.record("login", {"user": "admin"})
result = t.check_prerequisites("export", {}, ["login"])
assert result.satisfied, "Prereq should be satisfied"
print("PASS: StepTracker prerequisites")

# Test 8: Context
from hermes_forge.context.strategies import NoCompact

nc = NoCompact()
r, p = nc.compact(["a", "b"], 4096)
assert p == 0, "NoCompact should return phase 0"
print("PASS: Context strategies")

print("\n=== ALL FUNCTIONAL CHECKS PASSED ===")
