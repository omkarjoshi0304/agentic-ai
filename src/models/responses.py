"""Response models using Pydantic.

LEARNING NOTES:
- Response models document what the API returns.
- FastAPI uses these for OpenAPI docs AND for serialization.
- The `model_config` with `json_schema_extra` adds examples to the docs.
- Compare with LCS: src/models/responses.py
"""

from pydantic import BaseModel, Field


class KnowledgeChunk(BaseModel):
    """A piece of retrieved knowledge.

    LEARNING: In real RAG systems, documents are split into chunks,
    embedded as vectors, and stored in a vector database. When a query
    comes in, the most similar chunks are retrieved. Each chunk has a
    similarity score indicating how relevant it is.
    """

    content: str = Field(description="The text content of the knowledge chunk")
    source: str = Field(description="Where this knowledge came from")
    score: float = Field(description="Similarity score (0-1, higher = more relevant)")


class ChatResponse(BaseModel):
    """Response from the chat endpoint.

    LEARNING: In LCS, responses follow the OpenAI responses API format
    with streaming support via SSE. Our simplified version returns
    a single JSON response to keep things clear for learning.
    """

    conversation_id: str = Field(description="Conversation ID for follow-up messages")
    reply: str = Field(description="The assistant's response")
    knowledge_used: list[KnowledgeChunk] = Field(
        default_factory=list,
        description="Knowledge chunks that informed the response",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "conversation_id": "abc-123",
                    "reply": "Your pod is OOMKilled because...",
                    "knowledge_used": [
                        {
                            "content": "Memory limits were changed in v2.3...",
                            "source": "KB-4521",
                            "score": 0.92,
                        }
                    ],
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    service_name: str
    llama_stack_connected: bool


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    status_code: int
