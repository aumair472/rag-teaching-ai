"""
Conversation memory service module.

Maintains per-session conversation history for multi-turn interactions.
Uses in-memory storage with optional Redis persistence.
"""

import json
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class MemoryService:
    """
    Session-based conversation memory manager.

    Stores conversation turns (question-answer pairs) keyed by session ID.
    Uses an in-memory deque with a configurable maximum length.

    Optionally persists to Redis for cross-restart durability.

    Attributes:
        max_history: Maximum number of turns to retain per session.
        sessions: In-memory storage of conversation histories.
    """

    def __init__(
        self,
        max_history: Optional[int] = None,
        redis_client: Optional[object] = None,
    ) -> None:
        settings = get_settings()
        self.max_history = max_history or settings.max_conversation_history
        self.sessions: Dict[str, Deque[Tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )
        self._redis = redis_client  # optional Redis for persistence
        logger.info(
            "MemoryService initialized",
            extra={"max_history": self.max_history},
        )

    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
    ) -> None:
        """
        Add a conversation turn to a session.

        Args:
            session_id: The session identifier.
            question: The user's question.
            answer: The assistant's answer.
        """
        self.sessions[session_id].append((question, answer))
        logger.debug(
            "Turn added to memory",
            extra={
                "session_id": session_id,
                "history_length": len(self.sessions[session_id]),
            },
        )

        # Persist to Redis if available
        if self._redis is not None:
            try:
                key = f"memory:{session_id}"
                data = json.dumps(list(self.sessions[session_id]))
                self._redis.setex(key, 3600 * 24, data)  # 24h TTL
            except Exception as exc:
                logger.warning(
                    "Failed to persist memory to Redis",
                    extra={"error": str(exc)},
                )

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        """
        Retrieve conversation history for a session.

        Attempts in-memory lookup first, falling back to Redis.

        Args:
            session_id: The session identifier.

        Returns:
            A list of ``(question, answer)`` tuples.
        """
        if session_id in self.sessions:
            return list(self.sessions[session_id])

        # Try Redis fallback
        if self._redis is not None:
            try:
                key = f"memory:{session_id}"
                data = self._redis.get(key)
                if data:
                    turns = json.loads(data)
                    for q, a in turns:
                        self.sessions[session_id].append((q, a))
                    return list(self.sessions[session_id])
            except Exception as exc:
                logger.warning(
                    "Failed to load memory from Redis",
                    extra={"error": str(exc)},
                )

        return []

    def format_history(self, session_id: str) -> str:
        """
        Format conversation history as a string for the LLM prompt.

        Args:
            session_id: The session identifier.

        Returns:
            A formatted string of previous conversation turns.
        """
        history = self.get_history(session_id)
        if not history:
            return "No previous conversation."

        parts: List[str] = []
        for i, (question, answer) in enumerate(history, 1):
            parts.append(f"Turn {i}:")
            parts.append(f"  Student: {question}")
            parts.append(f"  Assistant: {answer}")

        return "\n".join(parts)

    def clear_session(self, session_id: str) -> None:
        """
        Clear conversation history for a specific session.

        Args:
            session_id: The session identifier.
        """
        if session_id in self.sessions:
            del self.sessions[session_id]

        if self._redis is not None:
            try:
                self._redis.delete(f"memory:{session_id}")
            except Exception:
                pass

        logger.info("Session cleared", extra={"session_id": session_id})

    def active_sessions(self) -> int:
        """Return the count of active sessions."""
        return len(self.sessions)
