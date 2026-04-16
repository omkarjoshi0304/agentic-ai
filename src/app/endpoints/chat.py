"""Chat endpoint — the core of the assistant.

LEARNING NOTES:
- This is our simplified version of LCS's /v1/responses endpoint.
- The flow mirrors the Lightspeed Stack architecture:
  1. Receive user message
  2. Search knowledge base (RAG retrieval)
  3. Build prompt with context
  4. Call LLM for inference
  5. Store conversation history
  6. Return response with knowledge references
- Compare with LCS: src/app/endpoints/responses.py

OPTION C (HYBRID) ARCHITECTURE:
- This endpoint handles the "shared knowledge" part (RAG, LLM inference)
- In the real architecture, oc/kubectl operations run LOCALLY on the
  developer's workstation via Goose's local MCP servers
- The LCS provides knowledge; the local agent provides cluster actions
"""

import logging

from fastapi import APIRouter

from models.requests import ChatRequest
from models.responses import ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Chat"])

SYSTEM_PROMPT = """You are an OpenShift assistant that helps developers diagnose and resolve
issues with their OpenShift clusters. You have access to a knowledge base of OpenShift
documentation, known issues, and troubleshooting guides.

When answering:
- Be specific and actionable
- Include relevant 'oc' commands the user can run
- Reference knowledge base articles when applicable
- If you're not sure, say so and suggest diagnostic steps
"""


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with the OpenShift assistant",
    description="Send a message and get an AI-powered response augmented with OpenShift knowledge.",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message through the RAG + LLM pipeline.

    LEARNING: This function orchestrates the full pipeline:
    1. Get or create conversation
    2. Retrieve relevant knowledge (RAG)
    3. Build augmented prompt
    4. Generate LLM response
    5. Store conversation turn
    6. Return response with metadata

    In LCS, this is much more complex — it handles streaming (SSE),
    tool calls, agent loops, quota checking, and moderation shields.
    But the core flow is the same.
    """
    from app.main import app_state

    conversation_store = app_state["conversation_store"]
    llm_client = app_state["llm_client"]
    config = app_state["config"]

    # Step 1: Get or create conversation
    if request.conversation_id and conversation_store.exists(request.conversation_id):
        conversation_id = request.conversation_id
    else:
        conversation_id = conversation_store.create_conversation()

    # Step 2: RAG retrieval — search knowledge base for relevant context
    knowledge_chunks = []
    context = ""
    if request.use_knowledge_base and config.knowledge_base.enabled:
        from services.knowledge import search_knowledge

        knowledge_chunks = search_knowledge(
            request.message,
            top_k=3,
            threshold=config.knowledge_base.similarity_threshold,
        )
        if knowledge_chunks:
            context = "\n\n".join(
                f"[{chunk.source}] (relevance: {chunk.score})\n{chunk.content}"
                for chunk in knowledge_chunks
            )
            logger.info("Found %d relevant knowledge chunks", len(knowledge_chunks))

    # Step 3: Get conversation history for multi-turn context
    history = conversation_store.get_history(conversation_id) or []

    # Step 4: Generate LLM response
    reply = await llm_client.generate(
        user_message=request.message,
        system_prompt=SYSTEM_PROMPT,
        context=context,
        conversation_history=history if history else None,
    )

    # Step 5: Store the conversation turn
    conversation_store.add_turn(conversation_id, request.message, reply)

    # Step 6: Return response with knowledge references
    return ChatResponse(
        conversation_id=conversation_id,
        reply=reply,
        knowledge_used=knowledge_chunks,
    )
