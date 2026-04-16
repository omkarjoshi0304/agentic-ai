"""LLM client service — wraps Llama Stack inference.

LEARNING NOTES:
- Llama Stack provides a unified API for LLM inference, agents, and tools.
- The real LCS uses `llama_stack_client.AsyncLlamaStackClient` (async HTTP client).
- Our version tries to connect to a real Llama Stack server, and falls back
  to a mock response if unavailable — so you can learn without running Llama Stack.
- Compare with LCS: src/client.py (AsyncLlamaStackClientHolder singleton)

LLAMA STACK ARCHITECTURE:
- Llama Stack server runs the model and exposes an API
- Client sends inference requests with messages (system + user + context)
- Server returns generated text
- In LCS, tools (RAG, MCP servers) are registered with Llama Stack
  so the LLM can call them during its reasoning loop
"""

import logging

import httpx

from models.config import LlamaStackConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for Llama Stack inference.

    LEARNING: This follows the Singleton-like pattern used in LCS's
    AsyncLlamaStackClientHolder. In production, you want one shared
    HTTP client with connection pooling, not a new client per request.
    """

    def __init__(self, config: LlamaStackConfig) -> None:
        self.config = config
        self._http_client = httpx.AsyncClient(
            base_url=config.url,
            timeout=config.timeout,
        )
        self._connected = False

    async def check_connection(self) -> bool:
        """Check if Llama Stack server is reachable."""
        try:
            response = await self._http_client.get("/v1/models")
            self._connected = response.status_code == 200
            if self._connected:
                logger.info("Connected to Llama Stack at %s", self.config.url)
            return self._connected
        except httpx.ConnectError:
            logger.warning(
                "Cannot connect to Llama Stack at %s — using mock responses",
                self.config.url,
            )
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def generate(
        self,
        user_message: str,
        system_prompt: str = "",
        context: str = "",
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate a response from the LLM.

        LEARNING: This mirrors how LCS builds the message array for Llama Stack:
        1. System prompt (sets the assistant's persona and rules)
        2. Context from RAG (injected knowledge)
        3. Conversation history (prior turns for multi-turn chat)
        4. Current user message

        Args:
            user_message: The current user question.
            system_prompt: Instructions for the LLM's behavior.
            context: Retrieved knowledge to ground the response.
            conversation_history: Prior messages for multi-turn context.

        Returns:
            The LLM's response text.
        """
        messages = self._build_messages(user_message, system_prompt, context, conversation_history)

        if self._connected:
            return await self._call_llama_stack(messages)
        return self._mock_response(user_message, context)

    def _build_messages(
        self,
        user_message: str,
        system_prompt: str,
        context: str,
        history: list[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        """Build the messages array for the LLM.

        LEARNING: The message format follows the chat completion standard:
        - role: "system" — instructions for the LLM
        - role: "user" — the human's messages
        - role: "assistant" — the LLM's prior responses
        """
        messages: list[dict[str, str]] = []

        # System prompt with optional RAG context
        full_system = system_prompt
        if context:
            full_system += (
                "\n\nUse the following knowledge base context to inform your answer. "
                "If the context is relevant, reference it. If not, rely on general knowledge.\n\n"
                f"CONTEXT:\n{context}"
            )
        if full_system:
            messages.append({"role": "system", "content": full_system})

        # Conversation history
        if history:
            messages.extend(history)

        # Current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    async def _call_llama_stack(self, messages: list[dict[str, str]]) -> str:
        """Call the real Llama Stack inference API.

        LEARNING: Llama Stack's /v1/chat/completions endpoint follows the
        OpenAI-compatible format. This is the same API that Goose and other
        tools can consume.
        """
        try:
            response = await self._http_client.post(
                "/v1/chat/completions",
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("Llama Stack inference failed: %s", e)
            return f"Error communicating with Llama Stack: {e}"

    def _mock_response(self, user_message: str, context: str) -> str:
        """Generate a mock response when Llama Stack is not available.

        LEARNING: This lets you run and test the full API flow without
        needing a running Llama Stack server. In a real deployment,
        you'd never use this — it's purely for development/learning.
        """
        if context:
            return (
                f"[Mock LLM Response — Llama Stack not connected]\n\n"
                f"Based on the knowledge base, here's what I found relevant "
                f"to your question about: '{user_message}'\n\n"
                f"Relevant context from knowledge base:\n{context}\n\n"
                f"In a real deployment with Llama Stack running, the LLM would "
                f"synthesize this context into a natural, detailed answer."
            )
        return (
            f"[Mock LLM Response — Llama Stack not connected]\n\n"
            f"You asked: '{user_message}'\n\n"
            f"No relevant knowledge base context was found. "
            f"With a real Llama Stack server, the LLM would answer from "
            f"its general training knowledge."
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http_client.aclose()
