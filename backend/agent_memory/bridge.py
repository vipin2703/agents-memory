"""
Chat ↔ Agent Memory bridge (orchestrator layer, stores nahi).

Read path for chat: the search tools (agent_tools) — the model calls them.
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
