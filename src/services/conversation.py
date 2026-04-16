"""Conversation history service.

LEARNING NOTES:
- Maintains multi-turn chat context so the LLM remembers prior messages.
- In the real LCS, conversations are stored in SQLite/PostgreSQL via
  the cache module (src/cache/). We use a simple in-memory dict.
- Conversation IDs let users continue a chat across multiple API calls.
- Compare with LCS: src/cache/ and src/a2a_storage/
"""

import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ConversationStore:
    """In-memory conversation history store.

    LEARNING: The real LCS uses multiple cache backends (in-memory, Redis,
    PostgreSQL) behind an abstract interface. This simplified version
    shows the core concept — storing and retrieving message history.
    """

    def __init__(self) -> None:
        # Dict of conversation_id -> list of messages
        self._conversations: dict[str, list[dict[str, str]]] = {}
        self._metadata: dict[str, dict] = {}

    def create_conversation(self) -> str:
        """Create a new conversation and return its ID."""
        conversation_id = str(uuid.uuid4())
        self._conversations[conversation_id] = []
        self._metadata[conversation_id] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "turn_count": 0,
        }
        logger.info("Created conversation %s", conversation_id)
        return conversation_id

    def get_history(self, conversation_id: str) -> list[dict[str, str]] | None:
        """Get the message history for a conversation.

        Returns None if the conversation doesn't exist.
        """
        return self._conversations.get(conversation_id)

    def add_turn(self, conversation_id: str, user_message: str, assistant_reply: str) -> None:
        """Add a user/assistant turn to the conversation history.

        LEARNING: Each 'turn' is a pair of messages (user + assistant).
        The LLM needs the full history to maintain context in multi-turn
        conversations. In production, you'd limit history length to avoid
        exceeding the model's context window.
        """
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []
            self._metadata[conversation_id] = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "turn_count": 0,
            }

        history = self._conversations[conversation_id]
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_reply})
        self._metadata[conversation_id]["turn_count"] += 1

        # Keep only last 10 turns (20 messages) to prevent context overflow
        max_messages = 20
        if len(history) > max_messages:
            self._conversations[conversation_id] = history[-max_messages:]

        logger.info(
            "Conversation %s: turn %d",
            conversation_id,
            self._metadata[conversation_id]["turn_count"],
        )

    def exists(self, conversation_id: str) -> bool:
        """Check if a conversation exists."""
        return conversation_id in self._conversations
