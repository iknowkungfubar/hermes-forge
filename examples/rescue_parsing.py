"""
Example 4: Rescue Parsing Showcase

Demonstrates all supported malformed tool call formats that Forge can rescue.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hermes_forge.guardrails.response_validator import rescue_tool_call


def main():
    print("=" * 60)
    print("Example 4: Rescue Parsing Showcase")
    print("=" * 60)

    valid_tools = {"get_weather", "send_email", "calculate", "search", "translate"}

    examples = [
        (
            "JSON in code fence",
            '```json {"name": "get_weather", "arguments": {"city": "Paris"}} ```',
        ),
        (
            "JSON in fence (no label)",
            '```\n{"name": "send_email", "arguments": {"to": "a@b.com", "subject": "Hi"}}\n```',
        ),
        ("Mistral [TOOL_CALLS] format", '[TOOL_CALLS] get_weather({"city": "London"})'),
        (
            "Mistral with reasoning prefix",
            'Let me check the weather first.\n[TOOL_CALLS] get_weather({"city": "Tokyo"})',
        ),
        ("Qwen XML format", '<tool_call>get_weather\n{"city": "Berlin"}\n</tool_call>'),
        ("Qwen with empty args", "<tool_call>search\n</tool_call>"),
        (
            "JSON array (multiple calls)",
            '```json [{"name": "get_weather", "arguments": {"city": "Rome"}}, {"name": "send_email", "arguments": {"to": "admin@test.com"}}] ```',
        ),
        (
            "Naked JSON object",
            '{"name": "calculate", "arguments": {"expression": "2+2"}}',
        ),
        (
            "Function-call format",
            '{"function": {"name": "translate", "arguments": {"text": "hello", "target": "fr"}}}',
        ),
    ]

    success = 0
    for name, text in examples:
        calls = rescue_tool_call(text, valid_tools)
        if calls:
            status = "✅"
            detail = f"→ {calls[0].tool}({calls[0].args})"
            if len(calls) > 1:
                detail += f" [+{len(calls) - 1} more]"
            success += 1
        else:
            status = "❌"
            detail = "→ (not rescued)"
        print(f"\n  {status} {name}")
        print(f"     {detail}")

    print(f"\n{'=' * 60}")
    print(f"Rescued: {success}/{len(examples)} formats")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
