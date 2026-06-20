"""
hermes-forge — A Hermes Agent MCP server and plugin bringing Forge's LLM
tool-calling guardrails to Hermes Agent.

This is a from-scratch reimplementation of the Forge guardrail framework
(https://github.com/antoinezambelli/forge) as a self-hosted MCP server,
Hermes plugin, and reusable skills. It provides:

- Response validation: Catch malformed or unknown tool calls before they
  reach execution.
- Rescue parsing: Extract structured tool calls from malformed LLM output
  (JSON in code fences, Mistral [TOOL_CALLS] format, Qwen XML).
- Step enforcement: Require certain tools be called before the terminal tool,
  with optional arg-matched prerequisites.
- Context management: Tiered compaction strategies that keep long-running
  workflows within the context budget.
- Proxy mode: A drop-in proxy that applies guardrails transparently to
  any OpenAI-compatible client.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("hermes-forge")
except PackageNotFoundError:
    __version__ = "0.1.0"

# Core
from hermes_forge.core.messages import (
    Message,
    MessageMeta,
    MessageRole,
    MessageType,
    ToolCallInfo,
)
from hermes_forge.core.workflow import (
    InferenceResult,
    LLMResponse,
    TextResponse,
    ToolCall,
    ToolDef,
    ToolSpec,
    Workflow,
)
from hermes_forge.core.runner import WorkflowRunner
from hermes_forge.core.steps import StepTracker

# Guardrails
from hermes_forge.guardrails.guardrails import (
    CheckResult,
    Guardrails,
)
from hermes_forge.guardrails.response_validator import ResponseValidator, ValidationResult
from hermes_forge.guardrails.step_enforcer import StepEnforcer, StepCheck
from hermes_forge.guardrails.error_tracker import ErrorTracker
from hermes_forge.guardrails.nudge import Nudge

# Context
from hermes_forge.context.manager import ContextManager, CompactEvent
from hermes_forge.context.strategies import (
    CompactStrategy,
    NoCompact,
    TieredCompact,
    SlidingWindowCompact,
)

# Prompts
from hermes_forge.prompts.templates import (
    build_tool_prompt,
    extract_tool_call,
    rescue_tool_call,
)
from hermes_forge.prompts.nudges import retry_nudge, step_nudge

# Tools
from hermes_forge.tools.respond import RESPOND_TOOL_NAME, respond_spec, respond_tool

# Errors
from hermes_forge.errors import (
    ForgeError,
    ToolCallError,
    ToolExecutionError,
    StepEnforcementError,
    PrerequisiteError,
    MaxIterationsError,
    BudgetResolutionError,
)

__all__ = [
    "Message",
    "MessageMeta",
    "MessageRole",
    "MessageType",
    "ToolCallInfo",
    "InferenceResult",
    "LLMResponse",
    "TextResponse",
    "ToolCall",
    "ToolDef",
    "ToolSpec",
    "Workflow",
    "WorkflowRunner",
    "StepTracker",
    "CheckResult",
    "Guardrails",
    "ResponseValidator",
    "ValidationResult",
    "StepEnforcer",
    "StepCheck",
    "ErrorTracker",
    "Nudge",
    "ContextManager",
    "CompactEvent",
    "CompactStrategy",
    "NoCompact",
    "TieredCompact",
    "SlidingWindowCompact",
    "build_tool_prompt",
    "extract_tool_call",
    "rescue_tool_call",
    "retry_nudge",
    "step_nudge",
    "RESPOND_TOOL_NAME",
    "respond_spec",
    "respond_tool",
    "ForgeError",
    "ToolCallError",
    "ToolExecutionError",
    "StepEnforcementError",
    "PrerequisiteError",
    "MaxIterationsError",
    "BudgetResolutionError",
]
