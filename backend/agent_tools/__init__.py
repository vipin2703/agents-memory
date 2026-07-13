"""
Agent tools — MCP-shaped registry.

Builtin memory tools:
  search_conversation (Elasticsearch exact chat)
  search_context (Neo4j graph)

Later: register MCP tool descriptors the same way + dispatch handlers.
"""

from .memory_tools import register_memory_tools
from .registry import ToolContext, ToolRegistry, get_tool_registry, tools_prompt_block

__all__ = [
    "ToolContext",
    "ToolRegistry",
    "get_tool_registry",
    "tools_prompt_block",
    "register_memory_tools",
]
