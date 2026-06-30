"""
WorkflowRunner — the agentic tool-calling loop.

Drives the LLM inference → validation → tool execution → repeat loop,
enforcing guardrails at each step. Wires together:
  - LLMClient (backend-agnostic inference)
  - ResponseValidator (tool-call schema & rescue)
  - StepEnforcer (required-steps & prerequisite ordering)
  - ErrorTracker (retry budgets)
  - ContextManager (token-budget compaction)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from hermes_forge.clients.base import LLMClient
from hermes_forge.core.inference import run_inference
from hermes_forge.core.messages import Message, MessageMeta, MessageRole, MessageType
from hermes_forge.core.workflow import ToolCall, ToolSpec, Workflow
from hermes_forge.errors import (
    MaxIterationsError,
    ToolCallError,
    ToolExecutionError,
)
from hermes_forge.guardrails.error_tracker import ErrorTracker
from hermes_forge.guardrails.response_validator import ResponseValidator
from hermes_forge.guardrails.step_enforcer import StepEnforcer
from hermes_forge.proxy.convert import forge_to_openai

logger = logging.getLogger(__name__)


class WorkflowRunner:
    """Executes a Workflow against tools with context management and guardrails.

    Usage:
        runner = WorkflowRunner(llm_client=my_client)
        result = await runner.run(workflow, "What's the weather?")

    The runner loops:
        1.  Compact messages if over budget
        2.  Send messages + tool specs to the LLM client
        3.  Validate the response (tool names, arg shape)
        4.  Enforce step ordering and prerequisites
        5.  Execute each valid tool call
        6.  Append tool results as new messages
        7.  Repeat until a terminal tool is called or max_iterations reached
    """

    def __init__(
        self,
        context_manager: Any | None = None,
        max_iterations: int = 10,
        max_retries_per_step: int = 3,
        max_tool_errors: int = 2,
        rescue_enabled: bool = True,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.context_manager = context_manager
        self.max_iterations = max_iterations
        self.max_retries_per_step = max_retries_per_step
        self.max_tool_errors = max_tool_errors
        self.rescue_enabled = rescue_enabled
        self._llm_client = llm_client

    # ── Client management ──────────────────────────────────────────

    def set_client(self, client: LLMClient) -> None:
        """Set or replace the LLM client used for inference."""
        self._llm_client = client

    # ── Main entry-point ───────────────────────────────────────────

    async def run(
        self,
        workflow: Workflow,
        user_message: str | None = None,
        prompt_vars: dict[str, str] | None = None,
        cancel_event: Any | None = None,
        *,
        messages: list[Message] | None = None,
    ) -> Any:
        """Execute the workflow and return the terminal tool's result.

        Args:
            workflow: The workflow definition.
            user_message: Input from the user (used when *messages* not
                          provided).
            prompt_vars: Variables for the system-prompt template.
            cancel_event: Optional asyncio.Event for cancellation.
            messages: Pre-built message list (alternative to
                      *user_message* + *prompt_vars*). Used by SlotWorker.

        Returns:
            The result from the terminal tool.

        Raises:
            MaxIterationsError: If *max_iterations* reached without a
                                terminal tool.
            ToolCallError: Retry budget exhausted.
            ToolExecutionError: Tool execution fails repeatedly.
            RuntimeError: No LLM client configured.
        """
        if self._llm_client is None:
            raise RuntimeError(
                "No LLM client configured. Pass llm_client to the "
                "constructor or call set_client() before running."
            )

        # --- Build initial messages if not provided directly ----------
        if messages is None:
            rendered = workflow.build_system_prompt(**(prompt_vars or {}))
            messages = [
                Message(
                    MessageRole.SYSTEM, rendered, MessageMeta(MessageType.SYSTEM_PROMPT)
                ),
                Message(
                    MessageRole.USER,
                    user_message or "",
                    MessageMeta(MessageType.USER_INPUT),
                ),
            ]

        tool_names = list(workflow.tools.keys())
        validator = ResponseValidator(tool_names, rescue_enabled=self.rescue_enabled)
        tool_prerequisites = {
            name: td.prerequisites
            for name, td in workflow.tools.items()
            if td.prerequisites
        }
        step_enforcer = StepEnforcer(
            required_steps=workflow.required_steps,
            terminal_tools=workflow.terminal_tools,
            tool_prerequisites=tool_prerequisites,
        )
        error_tracker = ErrorTracker(
            max_retries=self.max_retries_per_step,
            max_tool_errors=self.max_tool_errors,
        )

        iteration = 0
        terminal_result: Any | None = None

        # ── Main agentic loop ─────────────────────────────────────────
        while iteration < self.max_iterations:
            # Cancellation check
            if (
                cancel_event is not None
                and hasattr(cancel_event, "is_set")
                and cancel_event.is_set()
            ):
                logger.info(
                    "Workflow '%s' cancelled at iteration %d", workflow.name, iteration
                )
                break

            # Context compaction
            if self.context_manager is not None:
                should_compact, _ = self.context_manager.should_compact(messages)
                if should_compact:
                    messages = self.context_manager.compact(
                        messages,
                        step_hint=step_enforcer.summary_hint(),
                    )

            # Serialize messages and build tool definitions
            oai_messages = forge_to_openai(messages)
            oai_tools = (
                _build_openai_tools(workflow.get_tool_specs()) if tool_names else None
            )

            # ── Call the LLM ──────────────────────────────────────────
            try:
                response_data, token_usage = await self._llm_client.send(
                    oai_messages, oai_tools
                )
            except Exception as exc:
                logger.error("LLM call failed at iteration %d: %s", iteration, exc)
                raise ToolExecutionError("__llm__", cause=exc) from exc

            # ── Dispatch on response type ─────────────────────────────
            if not response_data:
                messages.append(_nudge("retry", "The LLM returned an empty response."))
                error_tracker.record_retry()
                if error_tracker.retries_exhausted:
                    raise ToolCallError("Empty LLM responses exhausted the retry limit")
                iteration += 1
                continue

            first = response_data[0]

            # Case 1 — structured tool-call response
            if isinstance(first, dict) and "tool" in first:
                tool_calls = [
                    ToolCall(tool=tc["tool"], args=tc.get("args", {}))
                    for tc in response_data
                ]
                result = await self._handle_tool_calls(
                    workflow,
                    tool_calls,
                    messages,
                    step_enforcer,
                    validator,
                    error_tracker,
                )
                if result is not None:
                    terminal_result = result
                    break

            # Case 2 — text / assistant response  (may embed a tool call)
            elif isinstance(first, dict) and (
                "content" in first or first.get("role") == "assistant"
            ):
                text = first.get("content", "")

                # Record the assistant message for context
                messages.append(
                    Message(
                        MessageRole.ASSISTANT,
                        text,
                        MessageMeta(MessageType.TEXT_RESPONSE),
                    )
                )

                if not text.strip():
                    messages.append(
                        _nudge("retry", "The LLM returned an empty response.")
                    )
                    error_tracker.record_retry()
                    if error_tracker.retries_exhausted:
                        raise ToolCallError(
                            "Empty LLM responses exhausted the retry limit"
                        )
                    iteration += 1
                    continue

                # Attempt rescue parsing (code fences, Qwen XML, …)
                inference_result = run_inference(text, tools=tool_names)

                if inference_result.needs_retry:
                    reason = (
                        inference_result.retry_reason
                        or "Please respond with a valid tool call."
                    )
                    messages.append(_nudge("retry", reason))
                    error_tracker.record_retry()
                    if error_tracker.retries_exhausted:
                        raise ToolCallError(f"Retry limit reached: {reason}")
                    iteration += 1
                    continue

                if inference_result.tool_calls:
                    # Rescue succeeded — treat as tool-call response
                    result = await self._handle_tool_calls(
                        workflow,
                        inference_result.tool_calls,
                        messages,
                        step_enforcer,
                        validator,
                        error_tracker,
                    )
                    if result is not None:
                        terminal_result = result
                        break
                else:
                    # Genuine plain-text response
                    if tool_names:
                        messages.append(
                            _nudge("retry", "Please use one of the available tools.")
                        )
                        error_tracker.record_retry()
                        if error_tracker.retries_exhausted:
                            raise ToolCallError(
                                "Text responses exhausted the retry limit"
                            )
                    else:
                        terminal_result = text
                        break

            # Case 3 — unrecognised format
            else:
                logger.warning(
                    "Unexpected LLM response format at iteration %d: %s",
                    iteration,
                    first,
                )
                messages.append(
                    _nudge("retry", "Unexpected response format from the LLM.")
                )
                error_tracker.record_retry()
                if error_tracker.retries_exhausted:
                    raise ToolCallError(
                        "Malformed LLM responses exhausted the retry limit"
                    )

            iteration += 1

        if terminal_result is not None:
            return terminal_result

        raise MaxIterationsError(
            self.max_iterations,
            step_enforcer.completed_steps,
            step_enforcer.pending(),
        )

    # ── Tool-call handling ──────────────────────────────────────────

    async def _handle_tool_calls(
        self,
        workflow: Workflow,
        tool_calls: list[ToolCall],
        messages: list[Message],
        step_enforcer: StepEnforcer,
        validator: ResponseValidator,
        error_tracker: ErrorTracker,
    ) -> Any | None:
        """Validate, order-enforce, and execute a batch of tool calls.

        Returns the terminal-tool result if one was called, else *None*.
        """
        # 1. Schema validation (tool name, argument shape)
        validation = validator.validate(tool_calls)
        if validation.needs_retry:
            nudge = validation.nudge
            kind = nudge.kind if nudge else "retry"
            content = (
                nudge.content if nudge else "Invalid tool call — please try again."
            )
            messages.append(_nudge(kind, content))
            error_tracker.record_retry()
            if error_tracker.retries_exhausted:
                raise ToolCallError("Invalid tool calls exhausted the retry limit")
            return None

        validated_calls = validation.tool_calls
        if not validated_calls:
            messages.append(_nudge("retry", "No valid tool calls found."))
            error_tracker.record_retry()
            return None

        # 2. Step-ordering check
        step_check = step_enforcer.check(validated_calls)
        if step_check.needs_nudge:
            if step_check.nudge is not None:
                messages.append(_nudge("step_enforcement", step_check.nudge.content))
            return None

        # 3. Prerequisite check
        prereq_check = step_enforcer.check_prerequisites(validated_calls)
        if prereq_check.needs_nudge:
            if prereq_check.nudge is not None:
                messages.append(_nudge("prerequisite_skip", prereq_check.nudge.content))
            return None

        # 4. Execute
        terminal_result: Any | None = None
        for tc in validated_calls:
            try:
                callable_fn = workflow.get_callable(tc.tool)
                args = tc.args if isinstance(tc.args, dict) else {}

                # Support sync *and* async callables
                if asyncio.iscoroutinefunction(callable_fn):
                    result = await callable_fn(**args)
                else:
                    result = callable_fn(**args)

                result_str = str(result) if result is not None else ""
                messages.append(
                    Message(
                        MessageRole.TOOL,
                        result_str,
                        MessageMeta(MessageType.TOOL_RESULT),
                        tool_name=tc.tool,
                    )
                )

                step_enforcer.record(tc.tool, args)
                error_tracker.reset_retries()
                error_tracker.record_result(success=True)

                if tc.tool in workflow.terminal_tools:
                    terminal_result = result

            except Exception as exc:
                logger.error("Tool '%s' execution failed: %s", tc.tool, exc)
                messages.append(
                    Message(
                        MessageRole.TOOL,
                        f"Error: {exc}",
                        MessageMeta(MessageType.TOOL_RESULT),
                        tool_name=tc.tool,
                    )
                )
                error_tracker.record_result(success=False)
                if error_tracker.tool_errors_exhausted:
                    raise ToolExecutionError(tc.tool, cause=exc) from exc

        return terminal_result


# ── Module-level helpers ──────────────────────────────────────────────


def _build_openai_tools(tool_specs: list[ToolSpec]) -> list[dict[str, Any]]:
    """Convert forge ``ToolSpec`` objects to OpenAI-compatible tool definitions."""
    tools: list[dict[str, Any]] = []
    for spec in tool_specs:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.get_json_schema(),
                },
            }
        )
    return tools


def _nudge(kind: str, content: str) -> Message:
    """Build a user-role nudge message that re-prompts the model."""
    return Message(
        MessageRole.USER,
        content,
        MessageMeta(MessageType.RETRY_NUDGE),
    )
