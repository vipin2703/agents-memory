"""
Builtin memory tools — three stores, clear jobs:

  WRITE (every turn, via persist_chat_turn — not tools):
    PostgreSQL  → episodic transcript + fact snapshot (source of truth log)
    Elasticsearch → full message text for exact / fuzzy conversation search
    Neo4j       → entities, facts, constraints, RELATES_TO (context graph)

  READ (tools the model chooses):
    search_conversation → Elasticsearch only (exact / past wording of chats)
    search_context      → Neo4j only (who/what/relations, long-term context)

Later MCP can register the same names or extras via ToolRegistry.
"""

from __future__ import annotations

import logging
from typing import Any

from .registry import ToolContext, ToolRegistry, ToolSpec

logger = logging.getLogger(__name__)

_QUERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search string (keywords or short phrase from the user question).",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


async def _search_conversation(ctx: ToolContext, args: dict[str, Any]) -> str:
    """Exact / past chat wording → Elasticsearch (not the full graph dump)."""
    from agent_memory.service import get_memory_service

    query = str(args.get("query") or "").strip()
    if not query:
        return "search_conversation: empty query — nothing searched."

    svc = get_memory_service()
    try:
        hits = await svc.search.search(
            user_id=ctx.user_id,
            query=query,
            session_id=None,  # all sessions for this user
            limit=8,
        )
    except Exception as e:
        logger.exception("search_conversation failed")
        return f"search_conversation ERROR: {e}"

    if not hits:
        return (
            f"search_conversation: no matching past messages for query={query!r}. "
            "Tell the user you don't have that conversation stored."
        )

    lines = [f"search_conversation (Elasticsearch) query={query!r}:"]
    for h in hits:
        score = f"{h.get('score'):.2f}" if h.get("score") is not None else "?"
        content = h.get("content") or ""
        if len(content) > 400:
            content = content[:400] + "…"
        role = h.get("role") or "?"
        sid = (h.get("session_id") or "")[:8]
        lines.append(f"  - ({role}, score={score}, session={sid}…) {content}")
    lines.append(
        "These are past chat lines. Quote or paraphrase only what answers the user."
    )
    return "\n".join(lines)


async def _search_context(ctx: ToolContext, args: dict[str, Any]) -> str:
    """Long-term facts / entities / relations → Neo4j graph."""
    from agent_memory.service import get_memory_service

    query = str(args.get("query") or "").strip()
    if not query:
        return "search_context: empty query — nothing searched."

    svc = get_memory_service()
    try:
        facts = await svc.graph.search_facts(
            user_id=ctx.user_id,
            query=query,
            limit=20,
            match_all_if_empty=False,
        )
        # Self / identity questions ("my name", "where do I work") rarely share
        # tokens with the stored facts — the name is an Entity node, not text that
        # literally contains "name". When the query matched nothing, fall back to
        # this user's whole known profile so the model can still answer.
        if not facts:
            facts = await svc.graph.facts_for_user(ctx.user_id, limit=20)
    except Exception as e:
        logger.exception("search_context failed")
        return f"search_context ERROR: {e}"

    if not facts:
        return (
            "search_context: nothing stored for this user yet. "
            "Tell the user you don't have that in context memory."
        )

    by_kind: dict[str, list[str]] = {}
    for f in facts:
        kind = f.get("kind") or "fact"
        text = f.get("text") or ""
        if text:
            by_kind.setdefault(kind, []).append(text)

    lines = [f"search_context (Neo4j graph) query={query!r}:"]
    for kind, items in by_kind.items():
        lines.append(f"  {kind}:")
        for t in items:
            lines.append(f"    - {t}")
    lines.append(
        "These are structured context notes (entities/facts/relations), not full chat logs. "
        "Use them for who/what/where style answers."
    )
    return "\n".join(lines)


def register_memory_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="search_conversation",
            description="Elasticsearch past chat lines (exact things user said). Not for hello.",
            input_schema=_QUERY_SCHEMA,
            handler=_search_conversation,
            source="builtin",
        )
    )
    registry.register(
        ToolSpec(
            name="search_context",
            description="Neo4j facts/entities/relations (name, city, links). Not for hello.",
            input_schema=_QUERY_SCHEMA,
            handler=_search_context,
            source="builtin",
        )
    )
