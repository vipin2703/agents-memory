"""
Chat ↔ Agent Memory bridge (orchestrator layer, stores nahi).

Structured chat se pehle recall, baad me user+assistant write.
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
            # "A -[PRED]-> B" → optional structured parse later; keep as fact string too
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
    svc = service or get_memory_service()
    errors: list[str] = []
    query = latest_user_text(messages)
    memory_block = ""
    graph_dict: dict[str, list[str]] = {
        "entities": [],
        "facts_about_user": [],
        "constraints": [],
    }
    recalled = False

    try:
        rec = await svc.recall(
            user_id=user_id,
            session_id=session_id,
            query=query or None,
            recent_limit=12,
            search_limit=8,
            graph_limit=40,
            include_recent_in_block=False,
        )
        memory_block = rec.memory_block or ""
        graph_dict = graph_facts_to_dict(rec.graph_facts)
        recalled = True
    except Exception as e:
        logger.warning("prepare_chat_memory recall failed: %s", e)
        errors.append(f"recall: {e}")

    memory_dict = merge_fact_dicts(client_memory, graph_dict)
    return {
        "memory_dict": memory_dict,
        "memory_block": memory_block,
        "errors": errors,
        "recalled": recalled,
    }


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
