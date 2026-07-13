"""
vLLM chat request/response models.
Memory / tools always ON on the backend — client does not toggle them.
"""

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class RelationTriple(BaseModel):
    """subject --predicate--> object (e.g. Rahul --LIVES_IN--> Pune)."""

    subject: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1)
    object: str = Field(..., min_length=1)


class ExtractedFacts(BaseModel):
    entities: list[str] = Field(default_factory=list, max_length=8)
    facts_about_user: list[str] = Field(default_factory=list, max_length=8)
    constraints: list[str] = Field(default_factory=list, max_length=8)
    relations: list[RelationTriple] = Field(default_factory=list, max_length=8)


class ChatRequest(BaseModel):
    """Client only sends messages (+ optional identity). No memory on/off switch."""

    messages: list[Message]
    # Optional identity for multi-user stores; if missing, backend assigns defaults.
    user_id: str | None = None
    session_id: str | None = None
    # Optional sampling — if omitted, server defaults apply via Field defaults.
    temperature: float = 0.7
    max_tokens: int = 4096


class MemoryStatus(BaseModel):
    enabled: bool = True
    recalled: bool = False
    wrote_user: bool = False
    wrote_assistant: bool = False
    tools_used: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# SSE final event: answer, extracted_facts, tools_used, memory_status, finish_reason
