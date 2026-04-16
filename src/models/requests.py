"""Request models using Pydantic.

LEARNING NOTES:
- These models define the shape of API request bodies.
- FastAPI automatically validates incoming JSON against these models.
- If validation fails, FastAPI returns a 422 with detailed error info.
- Compare with LCS: src/models/requests.py
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for the chat endpoint.

    LEARNING: In the real LCS, the /v1/responses endpoint accepts
    OpenAI-compatible request format. Our simplified version uses
    a simpler schema to focus on the core concepts.
    """

    message: str = Field(
        ...,  # ... means "required"
        min_length=1,
        max_length=4096,
        description="The user's message/question",
        examples=["Why is my pod crashlooping?"],
    )
    conversation_id: str | None = Field(
        default=None,
        description="Optional conversation ID for multi-turn chat",
    )
    use_knowledge_base: bool = Field(
        default=True,
        description="Whether to augment the response with knowledge base context",
    )


class KnowledgeSearchRequest(BaseModel):
    """Request body for knowledge base search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Search query for the knowledge base",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of results to return",
    )
