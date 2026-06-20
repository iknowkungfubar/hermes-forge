"""
Example 3: Context Compaction

Demonstrates how Forge manages context budgets in long-running workflows.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hermes_forge.core.messages import Message, MessageMeta, MessageRole, MessageType
from hermes_forge.context.manager import ContextManager
from hermes_forge.context.strategies import TieredCompact, SlidingWindowCompact, NoCompact


def main():
    print("=" * 60)
    print("Example 3: Context Compaction Strategies")
    print("=" * 60)

    # Build a simulated conversation history
    def build_conversation(turns=20):
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant.",
                    metadata=MessageMeta(MessageType.SYSTEM_PROMPT)),
            Message(role=MessageRole.USER, content="Help me analyze this dataset.",
                    metadata=MessageMeta(MessageType.USER_INPUT)),
        ]
        for i in range(turns):
            messages.append(
                Message(role=MessageRole.ASSISTANT, content=f"Step {i}: Processing data with very long response... " + "x" * 200,
                        metadata=MessageMeta(MessageType.TEXT_RESPONSE, step_index=i))
            )
            messages.append(
                Message(role=MessageRole.TOOL, content=f"Result {i}: " + "y" * 300,
                        metadata=MessageMeta(MessageType.TOOL_RESULT, step_index=i),
                        tool_name="process")
            )
        return messages

    messages = build_conversation(15)
    estimate = sum(len(m.content) for m in messages) // 4
    print(f"\n📊 Initial message count: {len(messages)}")
    print(f"📊 Estimated tokens: ~{estimate}")

    # TieredCompact
    print("\n--- Strategy 1: TieredCompact (keep_recent=2) ---")
    strategy = TieredCompact(keep_recent=2)
    compacted, phase = strategy.compact(messages, budget_tokens=2000)
    print(f"  After compaction: {len(compacted)} messages (phase {phase})")
    print(f"  Reduced by: {len(messages) - len(compacted)} messages")
    # Verify protected messages
    system_kept = any(m.role == MessageRole.SYSTEM for m in compacted)
    user_kept = any(m.role == MessageRole.USER for m in compacted)
    print(f"  System prompt preserved: {system_kept}")
    print(f"  User input preserved: {user_kept}")

    # SlidingWindowCompact
    print("\n--- Strategy 2: SlidingWindowCompact (keep_recent=3) ---")
    strategy2 = SlidingWindowCompact(keep_recent=3)
    compacted2, phase2 = strategy2.compact(messages, budget_tokens=2000)
    print(f"  After compaction: {len(compacted2)} messages (phase {phase2})")

    # NoCompact
    print("\n--- Strategy 3: NoCompact (passthrough) ---")
    strategy3 = NoCompact()
    compacted3, phase3 = strategy3.compact(messages, budget_tokens=2000)
    print(f"  After compaction: {len(compacted3)} messages (phase {phase3})")

    # ContextManager wrapper
    print("\n--- Strategy 4: ContextManager with TieredCompact ---")
    events = []
    cm = ContextManager(
        strategy=TieredCompact(keep_recent=2, compact_threshold=0.5),
        budget_tokens=2000,
        on_compact=lambda e: events.append(e),
    )

    should, tokens = cm.should_compact(messages)
    print(f"  Should compact: {should}")
    if should:
        compacted_cm = cm.compact(messages)
        print(f"  After CM compaction: {len(compacted_cm)} messages")
        print(f"  Compaction events fired: {len(events)}")

    print("\n" + "=" * 60)
    print("Example 3 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
