"""
Chat component for the Streamlit UI.

Renders the main chat interface using Streamlit's modern
chat_message API with streaming support, source citations,
and a voice agent (record audio → Whisper STT → RAG → TTS playback).
"""

import json
from typing import Any, Dict, List, Optional

import streamlit as st

from services.api_client import RAGApiClient


def render_chat(client: RAGApiClient) -> None:
    """
    Render the main chat interface.

    Displays conversation history from session state and handles
    new user input via typed text or voice recording.

    Args:
        client: The RAG API client instance.
    """
    # ── Initialize session state ────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Display conversation history ────────────────────────
    for msg in st.session_state.messages:
        _render_message(msg)

    # ── Input mode tabs: Text | Voice ───────────────────────
    text_tab, voice_tab = st.tabs(["💬 Text", "🎙️ Voice"])

    # ── Text input ──────────────────────────────────────────
    with text_tab:
        if prompt := st.chat_input(
            "Ask a question about the course material...",
            key="chat_input",
        ):
            _handle_text_question(client, prompt)

    # ── Voice input ─────────────────────────────────────────
    with voice_tab:
        _render_voice_input(client)


def _handle_text_question(client: RAGApiClient, prompt: str) -> None:
    """Add a typed question to the chat and generate a streaming response."""
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)

    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        _generate_response(client, prompt)


def _render_voice_input(client: RAGApiClient) -> None:
    """
    Render the voice agent UI panel.

    1. Browser mic recording via ``st.audio_input``.
    2. On submit: POST to ``/voice``, play back the MP3 answer,
       and show the transcription + sources in the chat.
    """
    st.markdown("#### 🎙️ Ask by Voice")
    st.caption(
        "Record your question. The assistant will transcribe it and speak the answer back."
    )

    session_id = st.session_state.get("session_id", "default")

    # st.audio_input is available from Streamlit ≥ 1.31
    audio_value = None
    try:
        audio_value = st.audio_input(
            "Click the microphone to record your question",
            key="voice_recorder",
        )
    except AttributeError:
        # Fallback for older Streamlit versions
        st.info("⬆️ Your Streamlit version doesn't support in-browser recording. Upload an audio file instead.")
        uploaded = st.file_uploader(
            "Upload audio (WAV / MP3 / M4A / WebM)",
            type=["wav", "mp3", "m4a", "webm", "ogg", "flac"],
            key="voice_uploader",
        )
        if uploaded is not None:
            audio_value = uploaded

    if audio_value is None:
        return

    # Show the recorded audio
    st.audio(audio_value, format="audio/wav")

    if st.button("🚀 Send to AI", key="voice_send"):
        audio_bytes = (
            audio_value.getvalue()
            if hasattr(audio_value, "getvalue")
            else audio_value.read()
        )
        filename = getattr(audio_value, "name", "recording.wav") or "recording.wav"

        with st.spinner("Transcribing and generating answer…"):
            mp3_bytes, metadata = client.voice_ask(
                audio_bytes=audio_bytes,
                filename=filename,
                session_id=session_id,
            )

        if "error" in metadata:
            st.error(f"⚠️ {metadata['error']}")
            return

        transcription: str = metadata.get("transcription", "")
        answer: str = metadata.get("answer", "")
        sources: list = metadata.get("sources", [])
        cached: bool = metadata.get("cached", False)
        latency: float = metadata.get("latency_ms", 0.0)

        # ── Add to chat history ─────────────────────────────
        st.session_state.messages.append({
            "role": "user",
            "content": f"🎙️ *{transcription}*",
        })
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "cached": cached,
            "latency_ms": latency,
        })

        # ── Display in chat ──────────────────────────────────
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(f"🎙️ *{transcription}*")

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(answer)
            if sources:
                _render_sources(sources)
            if cached:
                st.caption("⚡ Served from cache")
            st.caption(f"⏱️ {latency:.0f}ms")

        # ── Play audio answer ────────────────────────────────
        if mp3_bytes:
            st.success("🔊 Listen to the answer:")
            st.audio(mp3_bytes, format="audio/mp3", autoplay=True)


def _render_message(msg: Dict[str, Any]) -> None:
    """Render a single message from the conversation history."""
    role = msg["role"]
    avatar = "🧑‍🎓" if role == "user" else "🤖"

    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

        # Show sources if available
        if msg.get("sources"):
            _render_sources(msg["sources"])

        # Show metadata if available
        if msg.get("cached"):
            st.caption("⚡ Served from cache")
        if msg.get("latency_ms"):
            st.caption(f"⏱️ {msg['latency_ms']:.0f}ms")


def _generate_response(client: RAGApiClient, question: str) -> None:
    """
    Generate and stream the assistant's response.

    Attempts streaming first, falls back to standard request on failure.

    Args:
        client: The API client.
        question: The user's question.
    """
    session_id = st.session_state.get("session_id", "default")

    try:
        # ── Try streaming ───────────────────────────────────
        placeholder = st.empty()
        full_response = ""

        with st.spinner(""):
            for token in client.ask_stream(question, session_id):
                full_response += token
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

        # Store in session state
        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": full_response,
        }
        st.session_state.messages.append(assistant_msg)

    except Exception:
        # ── Fallback to non-streaming ───────────────────────
        with st.spinner("Thinking..."):
            response = client.ask(question, session_id)

        if "error" in response:
            error_msg = f"⚠️ {response['error']}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })
            return

        answer = response.get("answer", "No response received.")
        sources = response.get("sources", [])
        cached = response.get("cached", False)
        latency = response.get("latency_ms")

        st.markdown(answer)

        if sources:
            _render_sources(sources)
        if cached:
            st.caption("⚡ Served from cache")
        if latency:
            st.caption(f"⏱️ {latency:.0f}ms")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "cached": cached,
            "latency_ms": latency,
        })


def _render_sources(sources: List[Dict[str, Any]]) -> None:
    """
    Render source citations in an expandable section.

    Args:
        sources: List of source dicts from the API response.
    """
    with st.expander("📚 Sources", expanded=False):
        for i, src in enumerate(sources, 1):
            source_name = src.get("source_name", "Unknown")
            source_type = src.get("source_type", "").upper()
            snippet = src.get("text_snippet", "")
            sim_score = src.get("similarity_score", 0)
            rerank_score = src.get("rerank_score")

            # Location info
            location_parts: List[str] = []
            if src.get("page"):
                location_parts.append(f"Page {src['page']}")
            if src.get("slide"):
                location_parts.append(f"Slide {src['slide']}")
            if src.get("timestamp"):
                location_parts.append(f"⏱️ {src['timestamp']}")
            location = " · ".join(location_parts) if location_parts else ""

            # Render
            st.markdown(
                f"**[{i}] {source_name}** ({source_type})"
                + (f" — {location}" if location else "")
            )

            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"Similarity: {sim_score:.3f}")
            with col2:
                if rerank_score is not None:
                    st.caption(f"Re-rank: {rerank_score:.3f}")

            if snippet:
                st.text(snippet[:300])

            if i < len(sources):
                st.divider()
