"""
Tool registry (MCP-ready shape).

Each tool:
  name, description, input_schema (JSON Schema), source (builtin|mcp), handler

Model sees tools via tools_prompt_block(). Runtime dispatches by name.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

ToolHandler = Callable[["ToolContext", dict[str, Any]], Awaitable[str]]


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    source: str = "builtin"  # later: "mcp:<server>"


@dataclass
class ToolContext:
    """Per-request context for tool handlers (user/session/scopes)."""

    user_id: str
    session_id: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallResult:
    name: str
    arguments: dict[str, Any]
    result: str
    ok: bool = True
    error: str | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec
        logger.info("tool registered: %s (source=%s)", spec.name, spec.source)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def mcp_list_tools_payload(self) -> list[dict[str, Any]]:
        """Shape close to MCP tools/list result — ready for remote tools later."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
                "source": t.source,
            }
            for t in self._tools.values()
        ]

    async def execute(
        self, name: str, arguments: dict[str, Any] | str | None, ctx: ToolContext
    ) -> ToolCallResult:
        spec = self._tools.get(name)
        if not spec:
            return ToolCallResult(
                name=name,
                arguments={},
                result="",
                ok=False,
                error=f"unknown tool: {name}",
            )
        args = _coerce_args(arguments)
        try:
            text = await spec.handler(ctx, args)
            return ToolCallResult(name=name, arguments=args, result=text, ok=True)
        except Exception as e:
            logger.exception("tool %s failed", name)
            return ToolCallResult(
                name=name,
                arguments=args,
                result="",
                ok=False,
                error=str(e),
            )


def _coerce_args(arguments: dict[str, Any] | str | None) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        s = arguments.strip()
        if not s:
            return {}
        try:
            data = json.loads(s)
            return data if isinstance(data, dict) else {"value": data}
        except json.JSONDecodeError:
            return {"query": s}
    return {}


def tools_prompt_block(tools: list[ToolSpec]) -> str:
    """Short TOOLS block — long dumps make small models ignore the user message."""
    if not tools:
        return ""
    lines = [
        "TOOLS (call only when needed; hello = no tools):",
    ]
    for t in tools:
        # one line each — enough for the model, less pad fuel
        lines.append(f"- {t.name}: {t.description[:160]}")
    lines.append(
        'Call: {"name":"search_context","arguments":"{\\"query\\":\\"my name\\}"}'
    )
    return "\n".join(lines)


_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        from .memory_tools import register_memory_tools

        register_memory_tools(_registry)
    return _registry
