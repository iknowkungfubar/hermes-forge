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
- Backend lifecycle: Start/stop llama-server, Ollama, vLLM backends.
- Hardware detection: Auto-detect VRAM for context budget estimation.
- Client adapters: Native clients for Ollama, llamafile/llama-server, vLLM,
  and OpenAI-compatible backends.
- Inference loop: Compact, fold, serialize, validate, retry — shared by
  runner and proxy.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("hermes-forge")
except PackageNotFoundError:
    __version__ = "0.1.0"

from hermes_forge.clients.anthropic import AnthropicClient
from hermes_forge.clients.base import ChunkType, LLMClient, StreamChunk, TokenUsage
from hermes_forge.clients.llamafile import LlamafileClient
from hermes_forge.clients.ollama import OllamaClient
from hermes_forge.clients.openai_compat import OpenAICompatClient
from hermes_forge.clients.sampling_defaults import apply_sampling_defaults
from hermes_forge.clients.vllm import VLLMClient
from hermes_forge.context.hardware import HardwareProfile, detect_hardware
from hermes_forge.context.manager import CompactEvent, ContextManager
from hermes_forge.context.strategies import (
    CompactStrategy,
    NoCompact,
    SlidingWindowCompact,
    TieredCompact,
)
from hermes_forge.core.inference import run_inference
from hermes_forge.core.messages import (
    Message,
    MessageMeta,
    MessageRole,
    MessageType,
    ToolCallInfo,
)
from hermes_forge.core.reasoning import (
    DEFAULT_REASONING_REPLAY,
    REASONING_REPLAY_CHOICES,
    ReasoningReplay,
    filter_openai_reasoning_messages,
    validate_reasoning_replay,
)
from hermes_forge.core.runner import WorkflowRunner
from hermes_forge.core.slot_worker import SlotWorker
from hermes_forge.core.steps import StepTracker
from hermes_forge.core.workflow import (
    InferenceResult,
    LLMResponse,
    TextResponse,
    ToolCall,
    ToolDef,
    ToolSpec,
    Workflow,
)
from hermes_forge.errors import (
    BudgetResolutionError,
    ForgeError,
    MaxIterationsError,
    PrerequisiteError,
    StepEnforcementError,
    ToolCallError,
    ToolExecutionError,
)
from hermes_forge.guardrails.error_tracker import ErrorTracker
from hermes_forge.guardrails.guardrails import CheckResult, Guardrails
from hermes_forge.guardrails.nudge import Nudge
from hermes_forge.guardrails.response_validator import (
    ResponseValidator,
    ValidationResult,
    rescue_tool_call,
)
from hermes_forge.guardrails.step_enforcer import StepCheck, StepEnforcer
from hermes_forge.prompts.nudges import retry_nudge, step_nudge
from hermes_forge.prompts.templates import build_tool_prompt, extract_tool_call
from hermes_forge.proxy.proxy import ProxyServer
from hermes_forge.server import BudgetMode, ServerManager
from hermes_forge.tools.respond import RESPOND_TOOL_NAME, respond_spec, respond_tool

__all__ = [
    "DEFAULT_REASONING_REPLAY",
    "REASONING_REPLAY_CHOICES",
    # Tools
    "RESPOND_TOOL_NAME",
    "AnthropicClient",
    "BudgetMode",
    "BudgetResolutionError",
    # Guardrails
    "CheckResult",
    # Clients
    "ChunkType",
    "CompactEvent",
    "CompactStrategy",
    # Context
    "ContextManager",
    "ErrorTracker",
    # Errors
    "ForgeError",
    "Guardrails",
    "HardwareProfile",
    "InferenceResult",
    "LLMClient",
    "LLMResponse",
    "LlamafileClient",
    "MaxIterationsError",
    # Core
    "Message",
    "MessageMeta",
    "MessageRole",
    "MessageType",
    "NoCompact",
    "Nudge",
    "OllamaClient",
    "OpenAICompatClient",
    "PrerequisiteError",
    # Proxy & Server
    "ProxyServer",
    "ReasoningReplay",
    "ResponseValidator",
    "ServerManager",
    "SlidingWindowCompact",
    "SlotWorker",
    "StepCheck",
    "StepEnforcementError",
    "StepEnforcer",
    "StepTracker",
    "StreamChunk",
    "TextResponse",
    "TieredCompact",
    "TokenUsage",
    "ToolCall",
    "ToolCallError",
    "ToolCallInfo",
    "ToolDef",
    "ToolExecutionError",
    "ToolSpec",
    "VLLMClient",
    "ValidationResult",
    "Workflow",
    "WorkflowRunner",
    "apply_sampling_defaults",
    # Prompts
    "build_tool_prompt",
    "detect_hardware",
    "extract_tool_call",
    "filter_openai_reasoning_messages",
    "rescue_tool_call",
    "respond_spec",
    "respond_tool",
    "retry_nudge",
    "run_inference",
    "step_nudge",
    "validate_reasoning_replay",
]
