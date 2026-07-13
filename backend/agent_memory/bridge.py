"""
Chat ↔ Agent Memory bridge (orchestrator layer, stores nahi).

Read path for chat is via the search_memory TOOL (agent_tools), not auto-inject.
prepare_chat_memory remains for direct /memory API + debugging.
Write path: persist_chat_turn after generation.
"""

from __future__ import annotations

import logging
from typing import Any

from .schemas import ExtractedFacts, MemoryWriteRequest
from .service import AgentMemoryService, get_memory_service

logger = logging.getLogger(__name__)


def latest_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


def merge_fact_dicts(*parts: dict | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {
        "entities": [],
        "facts_about_user": [],
        "constraints": [],
    }
    for part in parts:
        if not part:
            continue
        for key in out:
            for item in part.get(key) or []:
                s = str(item).strip()
                if not s:
                    continue
                if s.lower() not in {x.lower() for x in out[key]}:
                    out[key].append(s)
    return out


def graph_facts_to_dict(graph_facts: list[Any]) -> dict:
    out: dict = {
        "entities": [],
        "facts_about_user": [],
        "constraints": [],
        "relations": [],
    }
    for g in graph_facts or []:
        kind = getattr(g, "kind", None) or (g.get("kind") if isinstance(g, dict) else "")
        text = getattr(g, "text", None) or (g.get("text") if isinstance(g, dict) else "")
        text = (text or "").strip()
        if not text:
            continue
        if kind == "entity":
            key = "entities"
        elif kind == "constraint":
            key = "constraints"
        elif kind == "relation":
            if text.lower() not in {x.lower() for x in out["facts_about_user"]}:
                out["facts_about_user"].append(text)
            continue
        else:
            key = "facts_about_user"
        if text.lower() not in {x.lower() for x in out[key]}:
            out[key].append(text)
    return out


async def prepare_chat_memory(
    *,
    user_id: str,
    session_id: str,
    messages: list[dict],
    client_memory: dict | None = None,
    service: AgentMemoryService | None = None,
) -> dict[str, Any]:
    """
    Always search stores with the user query. No intent keyword lists.

    memory_block is non-empty only when ES and/or Neo4j returned matches.
    """
    svc = service or get_memory_service()
    errors: list[str] = []
    query = latest_user_text(messages)
    memory_block = ""
    recalled = False
    hit_count = 0

    try:
        rec = await svc.recall(
            user_id=user_id,
            session_id=session_id,
            query=query or None,
            recent_limit=12,
            search_limit=8,
            graph_limit=20,
            # Do not dump session transcript — messages already in chat context.
            # Only DB search results (ES hits + query-matched graph).
            include_recent_in_block=False,
        )
        memory_block = (rec.memory_block or "").strip()
        hit_count = len(rec.search_hits or []) + len(rec.graph_facts or [])
        recalled = hit_count > 0
        logger.info(
            "prepare_chat_memory: user=%s hits es=%s graph=%s inject=%s q=%r",
            user_id,
            len(rec.search_hits or []),
            len(rec.graph_facts or []),
            bool(memory_block),
            (query or "")[:80],
        )
    except Exception as e:
        logger.warning("prepare_chat_memory recall failed: %s", e)
        errors.append(f"recall: {e}")

    return {
        "memory_dict": merge_fact_dicts(client_memory),
        "memory_block": memory_block,
        "errors": errors,
        "recalled": recalled,
        "hit_count": hit_count,
    }


async def recall_memory_block(
    *,
    user_id: str,
    session_id: str,
    messages: list[dict],
    service: AgentMemoryService | None = None,
) -> tuple[str, bool]:
    """
    Build a compact memory block to inject into the prompt EVERY turn so the
    model always has the user's long-term facts ready (name, city, employer,
    relations) even after those turns scroll out of the chat window.

    Two parts:
      1) Always: what we know about THIS user from the Neo4j graph.
      2) Query-based: related past chat lines from Elasticsearch.
    Returns (block, recalled).
    """
    svc = service or get_memory_service()
    query = latest_user_text(messages)
    sections: list[str] = []
    recalled = False

    try:
        facts = await svc.graph.facts_for_user(user_id, limit=40)
    except Exception as e:
        logger.warning("recall_memory_block graph: %s", e)
        facts = []
    if facts:
        by_kind: dict[str, list[str]] = {}
        for f in facts:
            kind = (f.get("kind") if isinstance(f, dict) else None) or "fact"
            text = (f.get("text") if isinstance(f, dict) else "") or ""
            text = text.strip()
            if text:
                by_kind.setdefault(kind, []).append(text)
        lines = ["KNOWN ABOUT THIS USER (long-term memory — trust and use this):"]
        for kind, items in by_kind.items():
            uniq = list(dict.fromkeys(items))
            if uniq:
                lines.append(f"  {kind}: " + "; ".join(uniq))
        sections.append("\n".join(lines))
        recalled = True

    if query:
        try:
            # role="user": recall the user's OWN past statements, never the
            # assistant's earlier (possibly wrong) replies.
            hits = await svc.search.search(
                user_id=user_id, query=query, session_id=None, role="user", limit=5
            )
        except Exception as e:
            logger.warning("recall_memory_block search: %s", e)
            hits = []
        if hits:
            lines = ["RELATED PAST MESSAGES (matched this question):"]
            for h in hits:
                content = (h.get("content") or "").strip()
                if len(content) > 200:
                    content = content[:200] + "…"
                if content:
                    lines.append(f"  - ({h.get('role') or '?'}) {content}")
            if len(lines) > 1:
                sections.append("\n".join(lines))
                recalled = True

    return ("\n\n".join(sections).strip(), recalled)


async def persist_chat_turn(
    *,
    user_id: str,
    session_id: str,
    user_text: str,
    assistant_text: str,
    extracted_facts: dict | ExtractedFacts | None,
    service: AgentMemoryService | None = None,
) -> dict[str, Any]:
    svc = service or get_memory_service()
    errors: list[str] = []
    wrote_user = wrote_assistant = False

    if isinstance(extracted_facts, ExtractedFacts):
        facts_model = extracted_facts
    elif isinstance(extracted_facts, dict):
        from .schemas import RelationTriple

        raw_rels = extracted_facts.get("relations") or []
        rels = []
        for r in raw_rels:
            if isinstance(r, dict) and r.get("subject") and r.get("object"):
                try:
                    rels.append(RelationTriple(**r))
                except Exception:
                    continue
        facts_model = ExtractedFacts(
            entities=list(extracted_facts.get("entities") or []),
            facts_about_user=list(extracted_facts.get("facts_about_user") or []),
            constraints=list(extracted_facts.get("constraints") or []),
            relations=rels,
        )
    else:
        facts_model = ExtractedFacts()

    if user_text:
        try:
            r = await svc.write_turn(
                MemoryWriteRequest(
                    user_id=user_id,
                    session_id=session_id,
                    role="user",
                    content=user_text,
                    extracted_facts=facts_model,
                )
            )
            wrote_user = r.sql_ok or r.search_ok or r.graph_ok
            errors.extend(r.errors)
        except Exception as e:
            logger.warning("persist user turn failed: %s", e)
            errors.append(f"write_user: {e}")

    if assistant_text:
        try:
            r = await svc.write_turn(
                MemoryWriteRequest(
                    user_id=user_id,
                    session_id=session_id,
                    role="assistant",
                    content=assistant_text,
                    extracted_facts=ExtractedFacts(),
                )
            )
            wrote_assistant = r.sql_ok or r.search_ok or r.graph_ok
            errors.extend(r.errors)
        except Exception as e:
            logger.warning("persist assistant turn failed: %s", e)
            errors.append(f"write_assistant: {e}")

    return {
        "wrote_user": wrote_user,
        "wrote_assistant": wrote_assistant,
        "errors": errors,
    }
