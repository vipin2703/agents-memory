"""Request/response models for agent memory API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RelationTriple(BaseModel):
    subject: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1)
    object: str = Field(..., min_length=1)


class ExtractedFacts(BaseModel):
    entities: list[str] = Field(default_factory=list, max_length=32)
    facts_about_user: list[str] = Field(default_factory=list, max_length=32)
    constraints: list[str] = Field(default_factory=list, max_length=32)
    relations: list[RelationTriple] = Field(default_factory=list, max_length=32)


class MemoryWriteRequest(BaseModel):
    """Ek turn teeno stores me likho: SQL + Elasticsearch + KG."""

    user_id: str = Field(..., min_length=1, description="Stable user key")
    session_id: str = Field(..., min_length=1, description="Chat session id")
    role: str = Field(..., description="user | assistant | system")
    content: str = Field(..., min_length=1)
    extracted_facts: ExtractedFacts | None = None
    message_id: str | None = None


class MemoryWriteResult(BaseModel):
    message_id: str
    session_id: str
    user_id: str
    sql_ok: bool
    search_ok: bool
    graph_ok: bool
    errors: list[str] = Field(default_factory=list)


class MemoryRecallRequest(BaseModel):
    user_id: str
    session_id: str | None = None
    query: str | None = Field(
        default=None,
        description="Elasticsearch pe full-text; empty = skip search",
    )
    recent_limit: int = Field(default=20, ge=1, le=100)
    search_limit: int = Field(default=10, ge=1, le=50)
    graph_limit: int = Field(default=30, ge=1, le=100)


class MemoryMessage(BaseModel):
    message_id: str
    session_id: str
    user_id: str
    role: str
    content: str
    created_at: datetime | None = None


class SearchHit(BaseModel):
    message_id: str
    session_id: str
    user_id: str
    role: str
    content: str
    score: float | None = None


class GraphFact(BaseModel):
    kind: str  # entity | fact_about_user | constraint | relation
    text: str
    user_id: str


class MemoryRecallResponse(BaseModel):
    recent_messages: list[MemoryMessage]
    search_hits: list[SearchHit]
    graph_facts: list[GraphFact]
    memory_block: str = Field(
        description="Ready-to-inject system memory text (SQL recent + KG + search)"
    )


class MemoryHealth(BaseModel):
    sql: dict[str, Any]
    search: dict[str, Any]
    graph: dict[str, Any]
