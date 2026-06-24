"""
Example 2: Step Enforcement Workflow

Demonstrates how Forge enforces required tool call ordering and
prerequisite satisfaction.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hermes_forge.core.workflow import ToolCall
from hermes_forge.guardrails.guardrails import Guardrails


def main():
    print("=" * 60)
    print("Example 2: Step Enforcement Workflow")
    print("=" * 60)

    # Simulate a multi-step data pipeline
    # Required: authenticate -> fetch_data -> analyze -> export (terminal)
    tools = ["authenticate", "fetch_data", "analyze", "export"]
    guard = Guardrails(
        tool_names=tools,
        terminal_tool="export",
        required_steps=["authenticate", "fetch_data", "analyze"],
    )

    print("\n📋 Workflow: authenticate → fetch_data → analyze → export")
    print("   (All steps required before export)")

    # Attempt 1: Try to export immediately (blocked)
    print("\n--- Attempt 1: Export without prerequisites ---")
    result = guard.check([ToolCall(tool="export", args={})])
    print(f"  Action: {result.action}")
    if result.action == "step_blocked":
        print(f"  Nudge: {result.nudge.content}")

    # Attempt 2: Call authenticate
    print("\n--- Attempt 2: Call authenticate ---")
    guard.record([("authenticate", {"user": "admin", "password": "***"})])
    print("  ✅ authenticate completed")
    print(f"  Pending: {guard._enforcer.pending()}")

    # Attempt 3: Skip fetch_data, try analyze
    print("\n--- Attempt 3: Skip to analyze ---")
    guard._errors.reset_retries()
    result = guard.check([ToolCall(tool="analyze", args={"query": "sales"})])
    if result.action == "execute":
        guard.record([("analyze", {"query": "sales"})])
    print(f"  Can proceed: {result.action == 'execute'}")
    if result.action != "execute":
        print(f"  Nudge: {result.nudge.content}")

    # Attempt 4: Follow the correct order
    print("\n--- Attempt 4: Correct order ---")
    guard._errors.reset_retries()
    result = guard.check([ToolCall(tool="fetch_data", args={"source": "db", "table": "orders"})])
    if result.action == "execute":
        guard.record([("fetch_data", {"source": "db", "table": "orders"})])
        print("  ✅ fetch_data completed")
    else:
        print(f"  ❌ Blocked: {result.nudge.content}")

    result = guard.check([ToolCall(tool="analyze", args={"query": "sales"})])
    if result.action == "execute":
        guard.record([("analyze", {"query": "sales"})])
        print("  ✅ analyze completed")
        print(f"  All required steps done: {guard._enforcer.is_satisfied()}")

    # Attempt 5: Now export should work
    print("\n--- Attempt 5: Export after prerequisites ---")
    result = guard.check([ToolCall(tool="export", args={"format": "pdf"})])
    print(f"  Can export: {result.action == 'execute'} ✅")

    print("\n" + "=" * 60)
    print("Example 2 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
