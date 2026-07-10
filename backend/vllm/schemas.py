# """
# vllm_service/schemas.py -- Chat related Pydantic models (sirf vLLM service ke liye).
# """

# from pydantic import BaseModel


# class Message(BaseModel):
#     role: str
#     content: str


# class ChatRequest(BaseModel):
#     messages: list[Message]
#     temperature: float = 0.7
#     max_tokens: int = 512


# class ChatResponse(BaseModel):
#     response: str




# """
# vllm_service/schemas.py -- Chat related Pydantic models (sirf vLLM service ke liye).
# """

# from pydantic import BaseModel, Field


# class Message(BaseModel):
#     role: str
#     content: str


# class ChatRequest(BaseModel):
#     messages: list[Message]
#     temperature: float = 0.9
#     max_tokens: int = 400


# class ChatResponse(BaseModel):
#     response: str


# class ExtractedFacts(BaseModel):
#     """Facts explicitly stated in THIS turn only. Empty list if none."""
#     entities: list[str] = Field(default=[], description="Names, companies, products, tools mentioned")
#     facts_about_user: list[str] = Field(default=[], description="Personal/background info about the user")
#     constraints: list[str] = Field(default=[], description="Limitations or boundaries stated")


# class StructuredChatOutput(BaseModel):
#     """
#     LLM ka guided-decoding output schema -- vLLM ko FORCE karta hai ki
#     response hamesha isi shape me aaye. Har field ki description khud
#     schema ke andar hai (Field(description=...)) -- isliye alag se lambi
#     system-prompt prose nahi likhni padti, instructions schema ke saath
#     hi vLLM ko milti hain.
#     """
#     answer: str
#     extracted_facts: ExtractedFacts



"""
vllm_service/schemas.py -- Chat related Pydantic models (sirf vLLM service ke liye).
"""

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int = 400


class ChatResponse(BaseModel):
    response: str


# NOTE: Class docstrings (""".."""} yaha jaanbujh ke NAHI likhe -- Pydantic
# automatically class-docstrings ko model_json_schema() ke "description"
# field me daal deta hai, jo phir LLM ko response_format ke saath bhej
# diya jaata hai. Dev-facing notes yaha # comments me hain (schema me
# nahi jaate), aur Field(description=...) jaanbujh ke rakhe hain kyunki
# wahi LLM-guidance ke liye the (attention-dilution fix, dekho client.py).

# ExtractedFacts: is turn me explicitly stated facts, categories me.

    # entities: list[str] = Field(default=[], description="Names, companies, products, tools mentioned")
    # facts_about_user: list[str] = Field(default=[], description="Personal/background info about the user")
    # constraints: list[str] = Field(default=[], description="Limitations or boundaries stated")

class ExtractedFacts(BaseModel):
    entities: list[str] = Field(default=[], description="Names, companies")
   



# StructuredChatOutput: guided-decoding output schema (answer + extracted_facts).
class StructuredChatOutput(BaseModel):
    answer: str
    extracted_facts: ExtractedFacts