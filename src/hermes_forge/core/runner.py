"""
WorkflowRunner — the agentic tool-calling loop.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from hermes_forge.core.messages import Message, MessageMeta, MessageRole, MessageType, ToolCallInfo
from hermes_forge.core.workflow import ToolCall, TextResponse, Workflow
from hermes_forge.errors import (
    MaxIterationsError,
    PrerequisiteError,
    StepEnforcementError,
    ToolCallError,
    ToolExecutionError,
)
from hermes_forge.guardrails.error_tracker import ErrorTracker
from hermes_forge.guardrails.response_validator import ResponseValidator
from hermes_forge.guardrails.step_enforcer import StepEnforcer


class WorkflowRunner:
    """Executes a Workflow against tools with context management and guardrails.

    Usage:
        runner = WorkflowRunner()
        result = await runner.run(workflow, "What's the weather?")
    """

    def __init__(
        self,
        context_manager: Any | None = None,
        max_iterations: int = 10,
        max_retries_per_step: int = 3,
        max_tool_errors: int = 2,
        rescue_enabled: bool = True,
    ) -> None:
        self.context_manager = context_manager
        self.max_iterations = max_iterations
        self.max_retries_per_step = max_retries_per_step
        self.max_tool_errors = max_tool_errors
        self.rescue_enabled = rescue_enabled

    async def run(
        self,
        workflow: Workflow,
        user_message: str,
        prompt_vars: dict[str, str] | None = None,
        cancel_event: Any | None = None,
    ) -> Any:
        """Execute the workflow and return the terminal tool's result.

        Args:
            workflow: The workflow definition.
            user_message: Input from the user.
            prompt_vars: Variables for the system prompt template.

        Returns:
            The result from the terminal tool.

        Raises:
            MaxIterationsError: If max_iterations exceeded.
            ToolCallError: If max_retries exhausted.
            ToolExecutionError: If tool execution fails repeatedly.
        """
        rendered_prompt = workflow.build_system_prompt(**(prompt_vars or {}))
        messages: list[Message] = []
        messages.append(Message(MessageRole.SYSTEM, rendered_prompt, MessageMeta(MessageType.SYSTEM_PROMPT)))
        messages.append(Message(MessageRole.USER, user_message, MessageMeta(MessageType.USER_INPUT)))

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

        tool_call_counter = 0
        iteration = 0
        terminal_result = None

        while iteration < self.max_iterations:
            if cancel_event is not None and hasattr(cancel_event, "is_set") and cancel_event.is_set():
                break

            # Compact if needed
            if self.context_manager is not None:
                should_compact, _ = self.context_manager.should_compact(messages)
                if should_compact:
                    messages = self.context_manager.compact(messages, step_hint=step_enforcer.summary_hint())

            # Build tool specs for the LLM
            tool_specs = workflow.get_tool_specs()
            tool_call_counter += 1

            # Simulate a call to the "LLM" by executing the workflow tools directly
            # In a real scenario, this would call an LLM client.
            # For the guardrail framework, we handle the response validation loop.
            response_text = f"[Simulated response for iteration {iteration}]"

            # For actual use, the caller would provide the LLM response here.
            # The guardrails validate and enforce step ordering.
            result_val = await self._execute_tools(
                workflow=workflow,
                messages=messages,
                step_enforcer=step_enforcer,
                validator=validator,
                error_tracker=error_tracker,
                tool_call_counter=tool_call_counter,
                iteration=iteration,
            )

            if result_val is not None:
                terminal_result = result_val
                break

            iteration += 1

        if terminal_result is not None:
            return terminal_result

        raise MaxIterationsError(
            self.max_iterations,
            step_enforcer.completed_steps,
            step_enforcer.pending(),
        )

    async def _execute_tools(
        self,
        workflow: Workflow,
        messages: list[Message],
        step_enforcer: StepEnforcer,
        validator: ResponseValidator,
        error_tracker: ErrorTracker,
        tool_call_counter: int,
        iteration: int,
    ) -> Any | None:
        """Execute the current batch of tool calls from the LLM response.

        In a real implementation, this would call the LLM, validate the
        response, and execute the returned tool calls.
        """
        # Placeholder — real LLM calls would happen here.
        # This is the integration point for the MCP server.
        return None
