"""
Neo4j knowledge graph — entities, facts, constraints, RELATIONS.

  (:User)-[:MENTIONED]->(:Entity)
  (:User)-[:HAS_FACT]->(:UserFact)
  (:User)-[:HAS_CONSTRAINT]->(:Constraint)
  (:Entity)-[:RELATES_TO {predicate, user_id}]->(:Entity)

Schema/constraints via neo4j-migrate (schema.cypher). App only MERGE/MATCH.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)


def _norm_pred(pred: str) -> str:
    p = re.sub(r"[^A-Za-z0-9]+", "_", (pred or "").strip()).strip("_")
    return (p or "RELATED_TO").upper()[:64]


class GraphStore:
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        self.uri = uri or os.getenv("MEMORY_NEO4J_URI", "bolt://neo4j:7687")
        self.user = user or os.getenv("MEMORY_NEO4J_USER", "neo4j")
        self.password = password or os.getenv("MEMORY_NEO4J_PASSWORD", "agentmemory")
        self._driver = None

    async def connect(self) -> None:
        if self._driver:
            return
        self._driver = AsyncGraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )
        async with self._driver.session() as session:
            await session.run("RETURN 1")
        logger.info("GraphStore connected (schema assumed present via neo4j-migrate)")

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def health(self) -> dict[str, Any]:
        try:
            await self.connect()
            assert self._driver
            async with self._driver.session() as session:
                rec = await session.run("RETURN 1 AS ok")
                row = await rec.single()
            return {"ok": True, "ping": row["ok"] if row else None, "uri": self.uri}
        except Exception as e:
            return {
                "ok": False,
                "error": str(e),
                "hint": "Run neo4j-migrate (schema.cypher) before app start",
            }

    async def upsert_facts(
        self,
        *,
        user_id: str,
        entities: list[str] | None = None,
        facts_about_user: list[str] | None = None,
        constraints: list[str] | None = None,
        relations: list[dict] | None = None,
    ) -> None:
        await self.connect()
        assert self._driver

        entities = [e.strip() for e in (entities or []) if e and e.strip()]
        facts = [f.strip() for f in (facts_about_user or []) if f and f.strip()]
        cons = [c.strip() for c in (constraints or []) if c and c.strip()]
        rels = relations or []

        # subjects/objects from relations bhi Entity banen
        for r in rels:
            if not isinstance(r, dict):
                continue
            for key in ("subject", "object"):
                name = str(r.get(key) or "").strip()
                if name and name not in entities:
                    entities.append(name)

        async with self._driver.session() as session:
            await session.run(
                "MERGE (u:User {id: $user_id}) ON CREATE SET u.created_at = datetime()",
                user_id=user_id,
            )
            for name in entities:
                await session.run(
                    """
                    MERGE (u:User {id: $user_id})
                    MERGE (e:Entity {user_id: $user_id, name: $name})
                    ON CREATE SET e.created_at = datetime()
                    SET e.updated_at = datetime()
                    MERGE (u)-[:MENTIONED]->(e)
                    """,
                    user_id=user_id,
                    name=name,
                )
            for text in facts:
                await session.run(
                    """
                    MERGE (u:User {id: $user_id})
                    MERGE (f:UserFact {user_id: $user_id, text: $text})
                    ON CREATE SET f.created_at = datetime()
                    SET f.updated_at = datetime()
                    MERGE (u)-[:HAS_FACT]->(f)
                    """,
                    user_id=user_id,
                    text=text,
                )
            for text in cons:
                await session.run(
                    """
                    MERGE (u:User {id: $user_id})
                    MERGE (c:Constraint {user_id: $user_id, text: $text})
                    ON CREATE SET c.created_at = datetime()
                    SET c.updated_at = datetime()
                    MERGE (u)-[:HAS_CONSTRAINT]->(c)
                    """,
                    user_id=user_id,
                    text=text,
                )

            # --- RELATIONS: (Entity)-[:RELATES_TO {predicate}]->(Entity) ---
            for r in rels:
                if not isinstance(r, dict):
                    continue
                sub = str(r.get("subject") or "").strip()
                obj = str(r.get("object") or "").strip()
                pred = _norm_pred(str(r.get("predicate") or ""))
                if not sub or not obj:
                    continue
                await session.run(
                    """
                    MERGE (u:User {id: $user_id})
                    MERGE (s:Entity {user_id: $user_id, name: $subject})
                    ON CREATE SET s.created_at = datetime()
                    SET s.updated_at = datetime()
                    MERGE (o:Entity {user_id: $user_id, name: $object})
                    ON CREATE SET o.created_at = datetime()
                    SET o.updated_at = datetime()
                    MERGE (u)-[:MENTIONED]->(s)
                    MERGE (u)-[:MENTIONED]->(o)
                    MERGE (s)-[rel:RELATES_TO {user_id: $user_id, predicate: $predicate}]->(o)
                    ON CREATE SET rel.created_at = datetime()
                    SET rel.updated_at = datetime()
                    """,
                    user_id=user_id,
                    subject=sub,
                    object=obj,
                    predicate=pred,
                )

    @staticmethod
    def _query_tokens(query: str) -> list[str]:
        """Tokens from user text for DB CONTAINS search — not an intent keyword list."""
        toks = re.findall(r"[\w']+", (query or "").lower(), flags=re.UNICODE)
        # keep short tokens too (city codes etc.) but drop pure 1-char noise
        return list(dict.fromkeys(t for t in toks if len(t) >= 2))

    async def facts_for_user(self, user_id: str, limit: int = 30) -> list[dict[str, Any]]:
        """Full dump (admin / debug). Chat path should use search_facts()."""
        return await self.search_facts(user_id=user_id, query="", limit=limit, match_all_if_empty=True)

    async def search_facts(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 30,
        match_all_if_empty: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Search Neo4j for this user with the query string.

        Empty query + match_all_if_empty=False → [] (do not dump whole graph on chat).
        Tokens matched via CONTAINS on entity names / fact texts / relation ends.
        """
        await self.connect()
        assert self._driver
        tokens = self._query_tokens(query)
        if not tokens and not match_all_if_empty:
            return []

        out: list[dict[str, Any]] = []
        async with self._driver.session() as session:
            if tokens:
                # Query-driven: only nodes whose name/text contains a query token
                rec = await session.run(
                    """
                    MATCH (u:User {id: $user_id})-[r]->(n)
                    WHERE any(t IN $tokens WHERE
                        (n.name IS NOT NULL AND toLower(toString(n.name)) CONTAINS t)
                        OR (n.text IS NOT NULL AND toLower(toString(n.text)) CONTAINS t)
                    )
                    RETURN type(r) AS rel_type, labels(n) AS labels,
                           n.name AS name, n.text AS text
                    LIMIT $limit
                    """,
                    user_id=user_id,
                    tokens=tokens,
                    limit=max(limit * 3, 30),
                )
            else:
                rec = await session.run(
                    """
                    MATCH (u:User {id: $user_id})-[r]->(n)
                    RETURN type(r) AS rel_type, labels(n) AS labels,
                           n.name AS name, n.text AS text
                    LIMIT $limit
                    """,
                    user_id=user_id,
                    limit=max(limit * 3, 30),
                )
            async for row in rec:
                rel_type = row["rel_type"] or ""
                labels = list(row["labels"] or [])
                name = row["name"]
                text = row["text"]
                if rel_type == "MENTIONED" or "Entity" in labels:
                    if name:
                        out.append(
                            {"kind": "entity", "text": name, "user_id": user_id}
                        )
                elif rel_type == "HAS_FACT" or "UserFact" in labels:
                    if text:
                        out.append(
                            {
                                "kind": "fact_about_user",
                                "text": text,
                                "user_id": user_id,
                            }
                        )
                elif rel_type == "HAS_CONSTRAINT" or "Constraint" in labels:
                    if text:
                        out.append(
                            {
                                "kind": "constraint",
                                "text": text,
                                "user_id": user_id,
                            }
                        )

            if tokens:
                rec = await session.run(
                    """
                    MATCH (s:Entity {user_id: $user_id})-[r]->(o:Entity {user_id: $user_id})
                    WHERE (type(r) = 'RELATES_TO' OR r.predicate IS NOT NULL)
                      AND any(t IN $tokens WHERE
                        toLower(toString(s.name)) CONTAINS t
                        OR toLower(toString(o.name)) CONTAINS t
                        OR toLower(toString(coalesce(r.predicate, ''))) CONTAINS t
                      )
                    RETURN s.name AS subject,
                           coalesce(r.predicate, type(r)) AS predicate,
                           o.name AS object
                    LIMIT $limit
                    """,
                    user_id=user_id,
                    tokens=tokens,
                    limit=limit,
                )
            else:
                rec = await session.run(
                    """
                    MATCH (s:Entity {user_id: $user_id})-[r]->(o:Entity {user_id: $user_id})
                    WHERE type(r) = 'RELATES_TO' OR r.predicate IS NOT NULL
                    RETURN s.name AS subject,
                           coalesce(r.predicate, type(r)) AS predicate,
                           o.name AS object
                    LIMIT $limit
                    """,
                    user_id=user_id,
                    limit=limit,
                )
            async for row in rec:
                sub = row["subject"] or ""
                pred = row["predicate"] or "RELATED_TO"
                obj = row["object"] or ""
                if sub and obj:
                    out.append(
                        {
                            "kind": "relation",
                            "text": f"{sub} -[{pred}]-> {obj}",
                            "user_id": user_id,
                        }
                    )

        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for item in out:
            key = f"{item['kind']}|{item['text']}".lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= limit * 2:
                break
        return deduped

    async def clear_user(self, user_id: str) -> None:
        await self.connect()
        assert self._driver
        async with self._driver.session() as session:
            await session.run(
                """
                MATCH (s:Entity {user_id: $user_id})-[r:RELATES_TO {user_id: $user_id}]->()
                DELETE r
                """,
                user_id=user_id,
            )
            await session.run(
                """
                MATCH (u:User {id: $user_id})
                OPTIONAL MATCH (u)-[r]->(n)
                DETACH DELETE u, n
                """,
                user_id=user_id,
            )
            await session.run(
                "MATCH (e:Entity {user_id: $user_id}) DETACH DELETE e",
                user_id=user_id,
            )
            await session.run(
                "MATCH (f:UserFact {user_id: $user_id}) DETACH DELETE f",
                user_id=user_id,
            )
            await session.run(
                "MATCH (c:Constraint {user_id: $user_id}) DETACH DELETE c",
                user_id=user_id,
            )
