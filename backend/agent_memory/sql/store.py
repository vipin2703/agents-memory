"""
Postgres = source of truth for agent memory.

Schema is NOT created here — apply via:
  docker compose service `memory-migrate`
  or: psql -f agent_memory/sql/schema.sql

App only opens a pool and runs DML.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

DEFAULT_DSN = "postgresql://postgresuser:postgres1938@postgres:5432/agent_memory"


def _memory_dsn() -> str:
    return os.getenv("MEMORY_DATABASE_URL") or os.getenv("DATABASE_URL") or DEFAULT_DSN


class SqlStore:
    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or _memory_dsn()
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self._pool:
            return
        # No CREATE DATABASE / CREATE TABLE — schema must already exist.
        self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)
        # Fail fast if tables missing
        async with self._pool.acquire() as conn:
            await conn.fetchval("SELECT 1 FROM sessions LIMIT 1")
        logger.info("SqlStore connected (schema assumed present)")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def health(self) -> dict[str, Any]:
        try:
            if not self._pool:
                await self.connect()
            assert self._pool
            async with self._pool.acquire() as conn:
                v = await conn.fetchval("SELECT 1")
            return {"ok": True, "ping": v}
        except Exception as e:
            return {
                "ok": False,
                "error": str(e),
                "hint": "Run memory-migrate (schema.sql) before app start",
            }

    async def ensure_session(self, session_id: str, user_id: str) -> None:
        assert self._pool
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sessions (id, user_id)
                VALUES ($1, $2)
                ON CONFLICT (id) DO NOTHING
                """,
                session_id,
                user_id,
            )

    async def append_message(
        self,
        *,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        message_id: str | None = None,
        entities: list[str] | None = None,
        facts_about_user: list[str] | None = None,
        constraints: list[str] | None = None,
        relations: list | None = None,
    ) -> dict[str, Any]:
        await self.connect()
        assert self._pool
        mid = message_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        await self.ensure_session(session_id, user_id)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO messages (id, session_id, user_id, role, content, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        role = EXCLUDED.role
                    """,
                    mid,
                    session_id,
                    user_id,
                    role,
                    content,
                    now,
                )
                await conn.execute(
                    """
                    INSERT INTO turn_facts (
                        message_id, session_id, user_id,
                        entities, facts_about_user, constraints, relations
                    )
                    VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb, $7::jsonb)
                    """,
                    mid,
                    session_id,
                    user_id,
                    json.dumps(entities or []),
                    json.dumps(facts_about_user or []),
                    json.dumps(constraints or []),
                    json.dumps(relations or []),
                )

        return {
            "message_id": mid,
            "session_id": session_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "created_at": now,
        }

    async def recent_messages(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        await self.connect()
        assert self._pool
        async with self._pool.acquire() as conn:
            if session_id:
                rows = await conn.fetch(
                    """
                    SELECT id, session_id, user_id, role, content, created_at
                    FROM messages
                    WHERE user_id = $1 AND session_id = $2
                    ORDER BY created_at DESC
                    LIMIT $3
                    """,
                    user_id,
                    session_id,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, session_id, user_id, role, content, created_at
                    FROM messages
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    user_id,
                    limit,
                )
        return [
            {
                "message_id": r["id"],
                "session_id": r["session_id"],
                "user_id": r["user_id"],
                "role": r["role"],
                "content": r["content"],
                "created_at": r["created_at"],
            }
            for r in reversed(list(rows))
        ]

    async def clear_session(self, *, user_id: str, session_id: str) -> None:
        await self.connect()
        assert self._pool
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM turn_facts WHERE user_id = $1 AND session_id = $2",
                user_id,
                session_id,
            )
            await conn.execute(
                "DELETE FROM messages WHERE user_id = $1 AND session_id = $2",
                user_id,
                session_id,
            )
            await conn.execute(
                "DELETE FROM sessions WHERE id = $1 AND user_id = $2",
                session_id,
                user_id,
            )

    async def clear_user(self, *, user_id: str) -> None:
        await self.connect()
        assert self._pool
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM turn_facts WHERE user_id = $1", user_id)
            await conn.execute("DELETE FROM messages WHERE user_id = $1", user_id)
            await conn.execute("DELETE FROM sessions WHERE user_id = $1", user_id)
