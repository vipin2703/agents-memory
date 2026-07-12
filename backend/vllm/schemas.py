"""
vllm_service/schemas.py -- Chat related Pydantic models (sirf vLLM service ke liye).
"""

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class RelationTriple(BaseModel):
    """subject --predicate--> object (e.g. Rahul --LIVES_IN--> Pune)."""

    subject: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1, description="Relation type, e.g. LIVES_IN, WORKS_AT")
    object: str = Field(..., min_length=1)


class ExtractedFacts(BaseModel):
    entities: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Names, companies, products, tools mentioned (max 8)",
    )
    facts_about_user: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Personal/background info about the user (max 8)",
    )
    constraints: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Limitations or boundaries stated (max 8)",
    )
    relations: list[RelationTriple] = Field(
        default_factory=list,
        max_length=8,
        description="Entity relations from user message (max 8)",
    )


class ChatRequest(BaseModel):
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int = 2048
    memory: ExtractedFacts | None = None
    user_id: str | None = Field(
        default=None, description="Stable user id for server memory stores"
    )
    session_id: str | None = Field(
        default=None, description="Chat session id for server memory stores"
    )
    use_agent_memory: bool = Field(
        default=True,
        description="If user_id+session_id set, use multi-store agent memory",
    )


class ChatResponse(BaseModel):
    response: str


class MemoryStatus(BaseModel):
    enabled: bool = False
    recalled: bool = False
    wrote_user: bool = False
    wrote_assistant: bool = False
    errors: list[str] = Field(default_factory=list)


class StructuredChatOutput(BaseModel):
    answer: str
    extracted_facts: ExtractedFacts
    memory_status: MemoryStatus | None = None
