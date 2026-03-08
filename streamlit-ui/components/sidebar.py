"""
Sidebar component for the Streamlit UI.

Renders session controls, health status, and document ingestion form.
"""

import uuid
from typing import Any, Dict, Optional

import streamlit as st

from services.api_client import RAGApiClient


def render_sidebar(client: RAGApiClient) -> None:
    """
    Render the sidebar with all controls and status indicators.

    Args:
        client: The RAG API client instance.
    """
    with st.sidebar:
        # ── Header ──────────────────────────────────────────
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
                <h2 style="margin:0; font-size:1.4rem;">🎓 RAG Assistant</h2>
                <p style="color:#888; font-size:0.85rem; margin-top:0.25rem;">
                    AI Teaching Assistant
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Health Status ───────────────────────────────────
        _render_health_status(client)

        st.divider()

        # ── Session Controls ────────────────────────────────
        _render_session_controls()

        st.divider()

        # ── Document Ingestion ──────────────────────────────
        _render_ingestion_form(client)

        st.divider()

        # ── Footer Links ────────────────────────────────────
        _render_footer(client)


def _render_health_status(client: RAGApiClient) -> None:
    """Display backend connection status."""
    st.markdown("##### System Status")

    health = client.health_check()
    status = health.get("status", "offline")

    if status == "healthy":
        st.success("🟢 Backend Connected", icon="✅")

        # Component details
        components = health.get("components", {})
        if components:
            cols = st.columns(len(components))
            for col, (name, state) in zip(cols, components.items()):
                with col:
                    is_healthy = "healthy" in state.lower()
                    icon = "✅" if is_healthy else "⚠️"
                    label = name.replace("_", " ").title()
                    st.caption(f"{icon} {label}")

        version = health.get("version", "N/A")
        st.caption(f"Version: {version}")
    else:
        st.error("🔴 Backend Offline", icon="❌")
        error = health.get("error", "")
        if error:
            st.caption(f"Error: {error[:100]}")


def _render_session_controls() -> None:
    """Render session management controls."""
    st.markdown("##### Session")

    # Initialize session ID
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]

    # Display session ID
    st.code(st.session_state.session_id, language=None)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 New Chat", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.rerun()

    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Message count
    msg_count = len(st.session_state.get("messages", []))
    st.caption(f"💬 {msg_count} messages")

    # Download conversation
    if msg_count > 0:
        import json
        conversation_data = json.dumps(
            st.session_state.messages, indent=2, default=str
        )
        st.download_button(
            "📥 Download Chat",
            data=conversation_data,
            file_name=f"chat_{st.session_state.session_id}.json",
            mime="application/json",
            use_container_width=True,
        )


def _render_ingestion_form(client: RAGApiClient) -> None:
    """Render the document ingestion form."""
    st.markdown("##### 📄 Ingest Document")

    with st.form("ingest_form", clear_on_submit=True):
        file_path = st.text_input(
            "File Path",
            placeholder="/path/to/document.pdf",
            help="Absolute path to the file on the server",
        )

        source_type = st.selectbox(
            "Source Type",
            options=["pdf", "ppt", "video"],
            index=0,
        )

        source_name = st.text_input(
            "Source Name (optional)",
            placeholder="e.g. Lecture 1 - Introduction",
            help="Human-readable name for the document",
        )

        submitted = st.form_submit_button(
            "📤 Ingest",
            use_container_width=True,
            type="primary",
        )

        if submitted:
            if not file_path:
                st.error("Please enter a file path.")
            else:
                with st.spinner("Ingesting document..."):
                    result = client.ingest(
                        file_path=file_path,
                        source_type=source_type,
                        source_name=source_name or None,
                    )

                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    chunks = result.get("chunks_created", 0)
                    name = result.get("source_name", "Unknown")
                    st.success(
                        f"✅ **{name}** ingested — {chunks} chunks created"
                    )


def _render_footer(client: RAGApiClient) -> None:
    """Render footer links and info."""
    st.markdown("##### Links")
    st.markdown(
        f"[📖 API Docs]({client.base_url}/docs) · "
        f"[📘 ReDoc]({client.base_url}/redoc)"
    )
    st.caption("Built with FastAPI + Streamlit")
