"""
AgentMemoryService — teeno alag stores ko orchestrate karta hai:

  sql/             Postgres
  elasticsearch/   Elasticsearch (BM25 + dense_vector)
  knowledge_graph/ Neo4j
"""

from __future__ import annotations

import logging
from typing import Any

from .elasticsearch import SearchStore
from .knowledge_graph import GraphStore
from .schemas import (
    ExtractedFacts,
    GraphFact,
    MemoryHealth,
    MemoryMessage,
    MemoryRecallResponse,
    MemoryWriteRequest,
    MemoryWriteResult,
    SearchHit,
)
from .sql import SqlStore

logger = logging.getLogger(__name__)

_service: "AgentMemoryService | None" = None


class AgentMemoryService:
    def __init__(
        self,
        sql: SqlStore | None = None,
        search: SearchStore | None = None,
        graph: GraphStore | None = None,
    ):
        self.sql = sql or SqlStore()
        self.search = search or SearchStore()
        self.graph = graph or GraphStore()

    async def startup(self) -> None:
        for name, store in (
            ("sql", self.sql),
            ("elasticsearch", self.search),
            ("knowledge_graph", self.graph),
        ):
            try:
                await store.connect()
                logger.info("agent_memory.%s connected", name)
            except Exception as e:
                logger.warning("agent_memory.%s connect failed: %s", name, e)

    async def shutdown(self) -> None:
        for store in (self.sql, self.search, self.graph):
            try:
                await store.close()
            except Exception:
                pass

    async def health(self) -> MemoryHealth:
        return MemoryHealth(
            sql=await self.sql.health(),
            search=await self.search.health(),
            graph=await self.graph.health(),
        )

    async def write_turn(self, req: MemoryWriteRequest) -> MemoryWriteResult:
        facts = req.extracted_facts or ExtractedFacts()
        errors: list[str] = []
        sql_ok = search_ok = graph_ok = False
        message_id = req.message_id or ""
        row: dict[str, Any] = {}

        rel_dump = [r.model_dump() if hasattr(r, "model_dump") else r for r in (facts.relations or [])]

        try:
            row = await self.sql.append_message(
                session_id=req.session_id,
                user_id=req.user_id,
                role=req.role,
                content=req.content,
                message_id=req.message_id,
                entities=facts.entities,
                facts_about_user=facts.facts_about_user,
                constraints=facts.constraints,
                relations=rel_dump,
            )
            message_id = row["message_id"]
            sql_ok = True
        except Exception as e:
            logger.exception("sql write failed")
            errors.append(f"sql: {e}")
            message_id = req.message_id or message_id or "unknown"

        try:
            await self.search.index_message(
                message_id=message_id if message_id != "unknown" else f"tmp-{req.session_id}",
                session_id=req.session_id,
                user_id=req.user_id,
                role=req.role,
                content=req.content,
                created_at=row.get("created_at"),
            )
            search_ok = True
        except Exception as e:
            logger.exception("elasticsearch write failed")
            errors.append(f"elasticsearch: {e}")

        try:
            await self.graph.upsert_facts(
                user_id=req.user_id,
                entities=facts.entities,
                facts_about_user=facts.facts_about_user,
                constraints=facts.constraints,
                relations=rel_dump,
            )
            graph_ok = True
        except Exception as e:
            logger.exception("knowledge_graph write failed")
            errors.append(f"knowledge_graph: {e}")

        return MemoryWriteResult(
            message_id=message_id,
            session_id=req.session_id,
            user_id=req.user_id,
            sql_ok=sql_ok,
            search_ok=search_ok,
            graph_ok=graph_ok,
            errors=errors,
        )

    async def recall(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
        query: str | None = None,
        recent_limit: int = 20,
        search_limit: int = 10,
        graph_limit: int = 30,
        include_recent_in_block: bool = True,
    ) -> MemoryRecallResponse:
        recent_raw: list[dict] = []
        search_raw: list[dict] = []
        graph_raw: list[dict] = []

        try:
            recent_raw = await self.sql.recent_messages(
                user_id=user_id, session_id=session_id, limit=recent_limit
            )
        except Exception as e:
            logger.warning("recall sql: %s", e)

        if query and query.strip():
            try:
                search_raw = await self.search.search(
                    user_id=user_id,
                    query=query.strip(),
                    session_id=None,
                    limit=search_limit,
                )
            except Exception as e:
                logger.warning("recall elasticsearch: %s", e)

        try:
            graph_raw = await self.graph.facts_for_user(user_id, limit=graph_limit)
        except Exception as e:
            logger.warning("recall knowledge_graph: %s", e)

        recent = [MemoryMessage(**m) for m in recent_raw]
        recent_ids = {m.message_id for m in recent}
        recent_texts = {(m.content or "").strip().lower() for m in recent}
        hits = []
        for h in search_raw:
            if h.get("message_id") in recent_ids:
                continue
            c = (h.get("content") or "").strip().lower()
            if c and c in recent_texts:
                continue
            hits.append(SearchHit(**h))
        gfacts = [GraphFact(**g) for g in graph_raw]
        block = _format_memory_block(
            recent if include_recent_in_block else [],
            hits,
            gfacts,
        )

        return MemoryRecallResponse(
            recent_messages=recent,
            search_hits=hits,
            graph_facts=gfacts,
            memory_block=block,
        )

    async def clear_session(self, user_id: str, session_id: str) -> dict[str, Any]:
        errors: list[str] = []
        sql_ok = search_ok = False
        try:
            await self.sql.clear_session(user_id=user_id, session_id=session_id)
            sql_ok = True
        except Exception as e:
            errors.append(f"sql: {e}")
        try:
            await self.search.delete_session(user_id=user_id, session_id=session_id)
            search_ok = True
        except Exception as e:
            errors.append(f"elasticsearch: {e}")
        return {"sql_ok": sql_ok, "search_ok": search_ok, "graph_kept": True, "errors": errors}

    async def clear_user(self, user_id: str) -> dict[str, Any]:
        errors: list[str] = []
        sql_ok = search_ok = graph_ok = False
        try:
            await self.sql.clear_user(user_id=user_id)
            sql_ok = True
        except Exception as e:
            errors.append(f"sql: {e}")
        try:
            await self.search.delete_user(user_id=user_id)
            search_ok = True
        except Exception as e:
            errors.append(f"elasticsearch: {e}")
        try:
            await self.graph.clear_user(user_id=user_id)
            graph_ok = True
        except Exception as e:
            errors.append(f"knowledge_graph: {e}")
        return {
            "sql_ok": sql_ok,
            "search_ok": search_ok,
            "graph_ok": graph_ok,
            "errors": errors,
        }


def _format_memory_block(
    recent: list[MemoryMessage],
    hits: list[SearchHit],
    gfacts: list[GraphFact],
) -> str:
    lines: list[str] = []

    if gfacts:
        lines.append("KNOWLEDGE GRAPH (stable long-term facts — treat as true):")
        by_kind: dict[str, list[str]] = {}
        for g in gfacts:
            by_kind.setdefault(g.kind, []).append(g.text)
        for kind, items in by_kind.items():
            lines.append(f"  {kind}:")
            for t in items:
                lines.append(f"    - {t}")
        lines.append("")

    if recent:
        lines.append("RECENT SESSION (exact transcript):")
        for m in recent:
            content = m.content if len(m.content) <= 500 else m.content[:500] + "…"
            lines.append(f"- {m.role}: {content}")
        lines.append("")

    if hits:
        lines.append("RELATED PAST (search — may be older sessions):")
        for h in hits:
            content = h.content if len(h.content) <= 400 else h.content[:400] + "…"
            score = f"{h.score:.2f}" if h.score is not None else "?"
            lines.append(f"- ({h.role}, score={score}) {content}")
        lines.append("")

    if not lines:
        return ""
    return (
        "MEMORY (use these; do not contradict known facts; "
        "prefer KNOWLEDGE GRAPH over fuzzy search hits):\n"
        + "\n".join(lines)
    ).strip()


def get_memory_service() -> AgentMemoryService:
    global _service
    if _service is None:
        _service = AgentMemoryService()
    return _service
