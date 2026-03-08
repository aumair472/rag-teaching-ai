"""
Session state manager.

Provides a clean abstraction over Streamlit's st.session_state
for managing conversation history, session IDs, and UI state.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st


class SessionManager:
    """
    Manages all session-level state for the RAG Teaching Assistant UI.

    Wraps ``st.session_state`` with typed accessors and default initialization.
    """

    # ── Keys ────────────────────────────────────────────────
    _SESSION_ID = "session_id"
    _MESSAGES = "messages"
    _DARK_MODE = "dark_mode"
    _METRICS = "last_metrics"
    _HEALTH = "last_health"
    _HEALTH_TS = "health_timestamp"

    @classmethod
    def initialize(cls) -> None:
        """Initialize all session state keys with defaults if absent."""
        defaults = {
            cls._SESSION_ID: str(uuid.uuid4())[:8],
            cls._MESSAGES: [],
            cls._DARK_MODE: True,
            cls._METRICS: {},
            cls._HEALTH: {},
            cls._HEALTH_TS: 0.0,
        }
        for key, default in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default

    # ── Session ID ──────────────────────────────────────────

    @classmethod
    def get_session_id(cls) -> str:
        """Return the current session ID."""
        return st.session_state.get(cls._SESSION_ID, "default")

    @classmethod
    def new_session(cls) -> str:
        """Generate a new session ID and clear messages."""
        new_id = str(uuid.uuid4())[:8]
        st.session_state[cls._SESSION_ID] = new_id
        st.session_state[cls._MESSAGES] = []
        st.session_state[cls._METRICS] = {}
        return new_id

    # ── Messages ────────────────────────────────────────────

    @classmethod
    def get_messages(cls) -> List[Dict[str, Any]]:
        """Return the conversation history."""
        return st.session_state.get(cls._MESSAGES, [])

    @classmethod
    def add_user_message(cls, content: str) -> None:
        """Append a user message."""
        cls.get_messages().append({
            "role": "user",
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })

    @classmethod
    def add_assistant_message(
        cls,
        content: str,
        sources: Optional[List[Dict]] = None,
        cached: bool = False,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Append an assistant message with optional metadata."""
        cls.get_messages().append({
            "role": "assistant",
            "content": content,
            "sources": sources or [],
            "cached": cached,
            "latency_ms": latency_ms,
            "timestamp": datetime.utcnow().isoformat(),
        })

    @classmethod
    def clear_messages(cls) -> None:
        """Clear conversation history."""
        st.session_state[cls._MESSAGES] = []

    @classmethod
    def message_count(cls) -> int:
        """Return the number of messages."""
        return len(cls.get_messages())

    # ── Dark Mode ───────────────────────────────────────────

    @classmethod
    def is_dark_mode(cls) -> bool:
        """Return dark mode state."""
        return st.session_state.get(cls._DARK_MODE, True)

    @classmethod
    def toggle_dark_mode(cls) -> bool:
        """Toggle dark mode and return new state."""
        current = cls.is_dark_mode()
        st.session_state[cls._DARK_MODE] = not current
        return not current

    # ── Metrics ─────────────────────────────────────────────

    @classmethod
    def set_metrics(cls, metrics: Dict[str, Any]) -> None:
        """Store the latest response metrics."""
        st.session_state[cls._METRICS] = metrics

    @classmethod
    def get_metrics(cls) -> Dict[str, Any]:
        """Return the latest metrics."""
        return st.session_state.get(cls._METRICS, {})

    # ── Health ──────────────────────────────────────────────

    @classmethod
    def set_health(cls, health: Dict[str, Any]) -> None:
        """Store health check result."""
        st.session_state[cls._HEALTH] = health
        st.session_state[cls._HEALTH_TS] = datetime.utcnow().timestamp()

    @classmethod
    def get_health(cls) -> Dict[str, Any]:
        """Return the last health check."""
        return st.session_state.get(cls._HEALTH, {})

    @classmethod
    def health_stale(cls, interval: int = 30) -> bool:
        """Check if health data is older than interval seconds."""
        ts = st.session_state.get(cls._HEALTH_TS, 0.0)
        return (datetime.utcnow().timestamp() - ts) > interval

    # ── Export / Import ─────────────────────────────────────

    @classmethod
    def export_conversation(cls) -> str:
        """Export conversation as formatted JSON."""
        data = {
            "session_id": cls.get_session_id(),
            "exported_at": datetime.utcnow().isoformat(),
            "messages": cls.get_messages(),
        }
        return json.dumps(data, indent=2, default=str)

    @classmethod
    def import_conversation(cls, json_str: str) -> bool:
        """
        Import conversation from JSON string.

        Returns:
            True if import succeeded.
        """
        try:
            data = json.loads(json_str)
            if "messages" in data:
                st.session_state[cls._MESSAGES] = data["messages"]
                if "session_id" in data:
                    st.session_state[cls._SESSION_ID] = data["session_id"]
                return True
            return False
        except (json.JSONDecodeError, KeyError):
            return False
