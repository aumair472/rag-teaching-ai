"""
Chat window component.

Renders the main chat interface with streaming support,
conversation history, and integrated RAG transparency panel.
"""

from typing import Any, Dict

import streamlit as st

from ui.components.message_renderer import render_message, render_sources
from ui.components.metrics_panel import render_inline_metrics
from ui.services.api_client import APIClient
from ui.services.streaming_client import StreamingClient
from ui.state.session_manager import SessionManager
from ui.utils.formatting import format_latency


def render_chat_window(
    api_client: APIClient,
    streaming_client: StreamingClient,
) -> None:
    """
    Render the full chat window.

    Displays conversation history and handles new user input
    with streaming-first, non-streaming fallback.

    Args:
        api_client: Standard API client for fallback.
        streaming_client: Streaming client for SSE responses.
    """
    # Render conversation history
    for msg in SessionManager.get_messages():
        render_message(msg)

    # Check for pending question from suggested questions
    pending_question = st.session_state.get("pending_question")
    if pending_question:
        # Clear it immediately to prevent re-processing
        st.session_state["pending_question"] = None
        
        # Add and display user message
        SessionManager.add_user_message(pending_question)
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(pending_question)

        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            _handle_response(api_client, streaming_client, pending_question)
        return

    # Chat input with placeholder
    if prompt := st.chat_input(
        "Ask a question about the course material… (e.g., 'What is gradient descent?')",
        key="main_chat_input",
    ):
        # Add and display user message
        SessionManager.add_user_message(prompt)
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            _handle_response(api_client, streaming_client, prompt)


def _handle_response(
    api_client: APIClient,
    streaming_client: StreamingClient,
    question: str,
) -> None:
    """
    Generate assistant response with streaming, falling back to standard.

    Args:
        api_client: Fallback API client.
        streaming_client: Primary streaming client.
        question: The user's question.
    """
    session_id = SessionManager.get_session_id()

    try:
        # ── Streaming (primary) ─────────────────────────────
        placeholder = st.empty()
        full_response = ""
        error_occurred = False

        for token in streaming_client.stream_response(question, session_id):
            full_response += token
            placeholder.markdown(full_response + " ▌")

            # Check for error markers
            if "⚠️" in token:
                error_occurred = True

        placeholder.markdown(full_response)

        if not error_occurred:
            SessionManager.add_assistant_message(content=full_response)

            # Try to get sources via non-streaming call quietly
            # (streaming doesn't return sources inline)
            _try_fetch_metadata(api_client, question, session_id)
        else:
            SessionManager.add_assistant_message(content=full_response)

    except Exception:
        # ── Fallback to non-streaming ───────────────────────
        _fallback_response(api_client, question, session_id)


def _fallback_response(
    api_client: APIClient,
    question: str,
    session_id: str,
) -> None:
    """Handle non-streaming fallback."""
    with st.spinner("Generating response…"):
        response = api_client.ask(question, session_id)

    if "error" in response:
        error_msg = f"⚠️ {response['error']}"
        st.error(error_msg)
        SessionManager.add_assistant_message(content=error_msg)
        return

    answer = response.get("answer", "No response received.")
    sources = response.get("sources", [])
    cached = response.get("cached", False)
    latency = response.get("latency_ms")

    st.markdown(answer)

    if sources:
        render_sources(sources)

    # Footer
    footer_parts = []
    if cached:
        footer_parts.append("⚡ Cached")
    if latency:
        footer_parts.append(f"⏱️ {format_latency(latency)}")
    if footer_parts:
        st.caption(" · ".join(footer_parts))

    SessionManager.add_assistant_message(
        content=answer,
        sources=sources,
        cached=cached,
        latency_ms=latency,
    )

    # Store metrics
    SessionManager.set_metrics({
        "latency_ms": latency,
        "cached": cached,
        "sources_count": len(sources),
        "client_latency_ms": response.get("_client_latency_ms"),
    })


def _try_fetch_metadata(
    api_client: APIClient,
    question: str,
    session_id: str,
) -> None:
    """
    Silently fetch response metadata (sources, latency)
    after a streaming response, without blocking the UI.
    """
    # The streaming response doesn't include sources, so we
    # update the last message if we can get them from a quiet call
    pass  # Sources come inline with stream; this is a hook for future enhancement
