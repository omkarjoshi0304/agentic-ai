"""Knowledge base search endpoint.

LEARNING NOTES:
- This exposes the RAG retrieval as a standalone endpoint.
- In LCS, RAG is used internally by the Llama Stack agent during inference.
- Having it as a separate endpoint lets you test and understand retrieval
  independently from the LLM generation step.
- Compare with LCS: src/app/endpoints/rags.py
"""

from fastapi import APIRouter, Depends

from middleware.auth import get_auth_dependency
from models.requests import KnowledgeSearchRequest
from models.responses import KnowledgeChunk

router = APIRouter(prefix="/v1/knowledge", tags=["Knowledge Base"])

# Auth dependency will be set during app startup
_auth_dependency = None


def set_auth_dependency(dep):
    global _auth_dependency
    _auth_dependency = dep


@router.post(
    "/search",
    response_model=list[KnowledgeChunk],
    summary="Search the knowledge base",
    description="Search OpenShift knowledge base for relevant information (RAG retrieval step).",
)
async def search_knowledge_base(request: KnowledgeSearchRequest) -> list[KnowledgeChunk]:
    """Search the knowledge base.

    LEARNING: This is the 'R' in RAG — Retrieval. The results from this
    endpoint are what get injected into the LLM prompt as context.
    Try different queries to see how the similarity matching works:
    - "pod crashlooping" → should match KB-001 (Pod Lifecycle)
    - "memory limit" → should match KB-002 (Resource Limits)
    - "expose service" → should match KB-003 (Routes)
    """
    from app.main import app_state
    from services.knowledge import search_knowledge

    threshold = app_state["config"].knowledge_base.similarity_threshold
    return search_knowledge(request.query, top_k=request.top_k, threshold=threshold)
