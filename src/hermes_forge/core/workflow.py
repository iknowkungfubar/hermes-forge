"""
Tool and workflow definitions.

Provides ToolSpec (what the LLM sees), ToolDef (binds spec to callable),
Workflow (declarative workflow definition), and ToolCall/TextResponse
(response types from the LLM).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, create_model


def _to_pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_")) + "Params"


def _json_schema_to_type(
    prop: dict[str, Any],
    field_name: str,
    model_name_prefix: str,
) -> type:
    if "enum" in prop:
        values = tuple(prop["enum"])
        return Literal[values]  # type: ignore[valid-type]
    json_type = prop.get("type", "string")
    type_map: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }
    if json_type in type_map:
        return type_map[json_type]
    if json_type == "object":
        sub_props = prop.get("properties", {})
        if sub_props:
            sub_required = set(prop.get("required", []))
            return _build_model(sub_props, sub_required, f"{model_name_prefix}_{field_name.capitalize()}")
        return dict
    if json_type == "array":
        items = prop.get("items", {})
        if items:
            item_type = _json_schema_to_type(items, field_name + "Item", model_name_prefix)
            return list[item_type]  # type: ignore[valid-type]
        return list
    return Any  # type: ignore[return-value]


def _build_model(
    properties: dict[str, Any],
    required: set[str],
    model_name: str,
) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for fname, fprop in properties.items():
        python_type = _json_schema_to_type(fprop, fname, model_name)
        description = fprop.get("description")
        default = fprop.get("default")
        if fname in required:
            if description:
                fields[fname] = (python_type, Field(description=description))
            else:
                fields[fname] = (python_type, ...)
        else:
            if default is not None:
                fields[fname] = (
                    python_type | None,
                    Field(default=default, description=description) if description else Field(default=default),
                )
            else:
                fields[fname] = (
                    python_type | None,
                    Field(default=None, description=description) if description else None,
                )
    return create_model(model_name, **fields)  # type: ignore[call-overload]


class ToolSpec(BaseModel):
    """Declarative tool schema — what the LLM sees."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    description: str
    parameters: type[BaseModel]

    @classmethod
    def from_json_schema(
        cls,
        name: str,
        description: str,
        schema: dict[str, Any],
    ) -> ToolSpec:
        """Create a ToolSpec from a raw JSON Schema dict."""
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        model_name = _to_pascal(name)
        params_cls = _build_model(properties, required, model_name)
        return cls(name=name, description=description, parameters=params_cls)

    def get_json_schema(self) -> dict[str, Any]:
        return self.parameters.model_json_schema()


@dataclass
class ToolDef:
    """Binds a tool schema to its implementation."""

    spec: ToolSpec
    callable: Callable[..., Any]
    prerequisites: list[str | dict[str, str]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.spec.name


@dataclass
class ToolCall:
    """Tool invocation returned by an LLM client."""
    tool: str
    args: Any
    reasoning: str | None = None


@dataclass
class TextResponse:
    """Non-tool-call response from the model."""
    content: str


LLMResponse = Union[list[ToolCall], TextResponse]


@dataclass
class InferenceResult:
    """Result from a single inference attempt (may include retries)."""
    response: LLMResponse
    attempts: int
    tool_call_counter: int
    new_messages: list  # Messages emitted during retries


@dataclass
class Workflow:
    """Declarative workflow definition."""

    name: str
    description: str
    tools: dict[str, ToolDef]
    required_steps: list[str]
    terminal_tool: str | list[str]
    system_prompt_template: str
    terminal_tools: frozenset[str] = field(default_factory=frozenset, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.terminal_tool, str):
            self.terminal_tools = frozenset([self.terminal_tool])
        else:
            self.terminal_tools = frozenset(self.terminal_tool)
        for key, tool_def in self.tools.items():
            if key != tool_def.name:
                raise ValueError(f"Tool key '{key}' does not match ToolDef name '{tool_def.name}'")
        tool_names = set(self.tools.keys())
        for step in self.required_steps:
            if step not in tool_names:
                raise ValueError(f"Required step '{step}' not in tools: {tool_names}")
        for tt in self.terminal_tools:
            if tt not in tool_names:
                raise ValueError(f"Terminal tool '{tt}' not in tools: {tool_names}")
            if tt in self.required_steps:
                raise ValueError(f"Terminal tool '{tt}' cannot also be a required step")

    def build_system_prompt(self, **kwargs: str) -> str:
        return self.system_prompt_template.format(**kwargs)

    def get_tool_specs(self) -> list[ToolSpec]:
        return [t.spec for t in self.tools.values()]

    def get_callable(self, tool_name: str) -> Callable[..., Any]:
        return self.tools[tool_name].callable
