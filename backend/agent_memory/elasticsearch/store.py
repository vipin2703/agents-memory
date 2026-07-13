"""
Elasticsearch store — BM25 + optional dense_vector.

Index schema is NOT created here — apply via docker `elasticsearch-migrate`
(index.json). App only indexes / searches documents.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from elasticsearch import AsyncElasticsearch, NotFoundError

logger = logging.getLogger(__name__)

INDEX_NAME = os.getenv("MEMORY_ELASTICSEARCH_INDEX", "agent_memory_messages")


class SearchStore:
    def __init__(
        self,
        hosts: list[str] | None = None,
        index: str | None = None,
    ):
        host = os.getenv("MEMORY_ELASTICSEARCH_URL", "http://elasticsearch:9200")
        self.hosts = hosts or [host]
        self.index = index or INDEX_NAME
        self._client: AsyncElasticsearch | None = None

    async def connect(self) -> None:
        if self._client:
            return
        self._client = AsyncElasticsearch(
            self.hosts,
            verify_certs=False,
            request_timeout=30,
        )
        exists = await self._client.indices.exists(index=self.index)
        if not exists:
            raise RuntimeError(
                f"Elasticsearch index '{self.index}' missing. "
                "Run docker compose service elasticsearch-migrate first."
            )
        logger.info("SearchStore connected (index=%s assumed present)", self.index)

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def health(self) -> dict[str, Any]:
        try:
            await self.connect()
            assert self._client
            info = await self._client.info()
            return {
                "ok": True,
                "cluster": info.get("cluster_name"),
                "version": (info.get("version") or {}).get("number"),
                "index": self.index,
            }
        except Exception as e:
            return {
                "ok": False,
                "error": str(e),
                "hint": "Run elasticsearch-migrate (index.json) before app start",
            }

    async def index_message(
        self,
        *,
        message_id: str,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        created_at: datetime | None = None,
        embedding: list[float] | None = None,
    ) -> None:
        await self.connect()
        assert self._client
        ts = created_at or datetime.now(timezone.utc)
        doc: dict[str, Any] = {
            "message_id": message_id,
            "session_id": session_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "created_at": ts.isoformat(),
        }
        if embedding is not None:
            doc["embedding"] = embedding

        await self._client.index(
            index=self.index,
            id=message_id,
            document=doc,
            refresh=True,
        )

    async def search(
        self,
        *,
        user_id: str,
        query: str,
        session_id: str | None = None,
        role: str | None = None,
        limit: int = 10,
        min_score: float | None = None,
        min_score_ratio: float = 0.35,
    ) -> list[dict[str, Any]]:
        """
        BM25 search over stored messages for this user.
        Weak / noise hits dropped via relative score floor (no intent hardcoding).
        Pass role="user" to recall only what the user themselves said (avoids
        echoing the assistant's own past — possibly wrong — replies).
        """
        if not (query or "").strip():
            return []
        await self.connect()
        assert self._client

        filters: list[dict[str, Any]] = [{"term": {"user_id": user_id}}]
        if session_id:
            filters.append({"term": {"session_id": session_id}})
        if role:
            filters.append({"term": {"role": role}})

        body: dict[str, Any] = {
            "size": limit,
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "content": {
                                    "query": query,
                                    "operator": "or",
                                }
                            }
                        }
                    ],
                    "filter": filters,
                }
            },
        }
        if min_score is not None:
            body["min_score"] = min_score

        try:
            res = await self._client.search(index=self.index, body=body)
        except NotFoundError:
            return []

        raw_hits = res.get("hits", {}).get("hits", []) or []
        if not raw_hits:
            return []

        top = float(raw_hits[0].get("_score") or 0.0)
        # Relative floor: keep hits that are actually competitive with the best match
        floor = max(0.5, top * min_score_ratio) if top > 0 else 0.5
        if min_score is not None:
            floor = max(floor, min_score)

        hits = []
        for h in raw_hits:
            score = float(h.get("_score") or 0.0)
            if score < floor:
                continue
            src = h.get("_source") or {}
            hits.append(
                {
                    "message_id": src.get("message_id") or h.get("_id"),
                    "session_id": src.get("session_id", ""),
                    "user_id": src.get("user_id", user_id),
                    "role": src.get("role", ""),
                    "content": src.get("content", ""),
                    "score": score,
                }
            )
        return hits

    async def delete_session(self, *, user_id: str, session_id: str) -> None:
        await self.connect()
        assert self._client
        await self._client.delete_by_query(
            index=self.index,
            query={
                "bool": {
                    "filter": [
                        {"term": {"user_id": user_id}},
                        {"term": {"session_id": session_id}},
                    ]
                }
            },
            refresh=True,
            conflicts="proceed",
        )

    async def delete_user(self, *, user_id: str) -> None:
        await self.connect()
        assert self._client
        await self._client.delete_by_query(
            index=self.index,
            query={"term": {"user_id": user_id}},
            refresh=True,
            conflicts="proceed",
        )
