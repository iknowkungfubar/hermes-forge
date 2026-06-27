"""
Example 1: Basic Tool Call Validation

Demonstrates the core guardrail feature: validating tool calls against
known tool schemas before execution.
"""

import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hermes_forge.guardrails.response_validator import ResponseValidator, rescue_tool_call
from hermes_forge.core.workflow import ToolCall, TextResponse


def main():
    print("=" * 60)
    print("Example 1: Basic Tool Call Validation")
    print("=" * 60)

    # Define available tools
    tools = ["get_weather", "send_email", "calculate"]
    validator = ResponseValidator(tool_names=tools)

    # Test 1: Valid tool call
    print("\n--- Test 1: Valid tool call ---")
    result = validator.validate([ToolCall(tool="get_weather", args={"city": "London"})])
    print("  Tool: get_weather(city='London')")
    print(f"  Valid: {not result.needs_retry} {'✅' if not result.needs_retry else '❌'}")

    # Test 2: Unknown tool
    print("\n--- Test 2: Unknown tool ---")
    result = validator.validate([ToolCall(tool="delete_database", args={})])
    print("  Tool: delete_database()")
    print(f"  Valid: {not result.needs_retry} {'✅' if not result.needs_retry else '❌'}")
    if result.needs_retry:
        print(f"  Error: {result.nudge.content if result.nudge else 'unknown error'}")

    # Test 3: Malformed args
    print("\n--- Test 3: Malformed args ---")
    result = validator.validate([ToolCall(tool="get_weather", args="not_a_dict")])
    print("  Tool: get_weather(args='not_a_dict')")
    print(f"  Valid: {not result.needs_retry} {'✅' if not result.needs_retry else '❌'}")
    if result.needs_retry:
        print(f"  Error: {result.nudge.content if result.nudge else 'malformed'}")

    # Test 4: Rescue parsing (JSON in code fence)
    print("\n--- Test 4: Rescue parsing ---")
    malformed = '```json {"name": "send_email", "arguments": {"to": "user@example.com", "subject": "Hello"}} ```'
    rescued = rescue_tool_call(malformed, set(tools))
    if rescued:
        print(f"  Input: {malformed[:60]}...")
        print(f"  Rescued: ✅ → {rescued[0].tool}({json.dumps(rescued[0].args)})")
    else:
        print("  Rescued: ❌")

    # Test 5: Text response
    print("\n--- Test 5: Text response handling ---")
    result = validator.validate(TextResponse(content="I'll help you with that."))
    print("  Response: 'I'll help you with that.'")
    print(f"  Needs retry: {result.needs_retry}")

    print("\n" + "=" * 60)
    print("Example 1 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
