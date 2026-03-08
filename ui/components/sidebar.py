"""
Sidebar component.

Renders all sidebar sections: brand, health status, session controls,
ingestion panel, metrics panel, and footer. Enhanced with better UX.
"""

from typing import Any, Dict

import streamlit as st

from ui.components.ingestion_panel import render_ingestion_panel
from ui.components.metrics_panel import render_metrics_panel
from ui.config import config
from ui.services.api_client import APIClient
from ui.state.session_manager import SessionManager


def render_sidebar(api_client: APIClient) -> None:
    """
    Render the complete sidebar.

    Sections:
        1. Brand header
        2. System health
        3. Session controls
        4. Document ingestion
        5. RAG metrics
        6. Quick tips
        7. Footer

    Args:
        api_client: API client for health checks and ingestion.
    """
    with st.sidebar:
        _render_brand()
        st.divider()
        _render_health(api_client)
        st.divider()
        _render_session_controls()
        st.divider()
        render_ingestion_panel(api_client)
        st.divider()
        _render_rag_panel()
        st.divider()
        _render_quick_tips()
        st.divider()
        _render_footer()


def _render_brand() -> None:
    """Render the sidebar brand header."""
    st.markdown(
        """<div class="sidebar-brand">
            <h2>🎓 RAG Assistant</h2>
            <p>AI Teaching Assistant</p>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_health(api_client: APIClient) -> None:
    """Render backend health status with component breakdown."""
    st.markdown("##### 🔗 System Status")

    # Refresh health if stale
    if SessionManager.health_stale(config.HEALTH_CHECK_INTERVAL):
        health = api_client.health()
        SessionManager.set_health(health)
    else:
        health = SessionManager.get_health()

    status = health.get("status", "offline")
    latency = health.get("latency_ms", 0)

    if status == "healthy":
        st.markdown(
            f'<span class="status-pill status-online">● Connected ({latency}ms)</span>',
            unsafe_allow_html=True,
        )

        # Component grid
        components = health.get("components", {})
        if components:
            cols = st.columns(len(components))
            for col, (name, state) in zip(cols, components.items()):
                with col:
                    is_ok = "healthy" in str(state).lower()
                    icon = "✅" if is_ok else "⚠️"
                    label = name.replace("_", " ").title()
                    st.caption(f"{icon} {label}")

        version = health.get("version", "")
        if version:
            st.caption(f"📦 v{version}")
    else:
        st.markdown(
            '<span class="status-pill status-offline">● Offline</span>',
            unsafe_allow_html=True,
        )
        error = health.get("error", "")
        if error:
            st.caption(f"_{error[:80]}_")
        st.info("💡 Start the backend: `uvicorn app.main:app --reload`")

    # Manual refresh
    if st.button("🔄 Refresh Status", key="health_refresh", use_container_width=True):
        health = api_client.health()
        SessionManager.set_health(health)
        st.rerun()


def _render_session_controls() -> None:
    """Render session management controls."""
    st.markdown("##### 💬 Conversation")

    # Session ID
    session_id = SessionManager.get_session_id()
    st.code(session_id, language=None)

    # Action buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✨ New Chat", use_container_width=True, type="primary"):
            SessionManager.new_session()
            st.rerun()

    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            SessionManager.clear_messages()
            st.rerun()

    # Stats
    count = SessionManager.message_count()
    st.caption(f"📊 {count} messages in session")

    # Export
    if count > 0:
        st.download_button(
            "📥 Export Chat",
            data=SessionManager.export_conversation(),
            file_name=f"rag_chat_{session_id}.json",
            mime="application/json",
            use_container_width=True,
        )

    # Import
    with st.expander("📤 Import Conversation", expanded=False):
        uploaded = st.file_uploader(
            "Upload JSON",
            type=["json"],
            key="import_file",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            content = uploaded.read().decode("utf-8")
            if SessionManager.import_conversation(content):
                st.success("✅ Imported successfully!")
                st.rerun()
            else:
                st.error("❌ Invalid format")


def _render_rag_panel() -> None:
    """Render the RAG transparency / metrics panel."""
    st.markdown("##### 📊 RAG Pipeline")
    render_metrics_panel()


def _render_quick_tips() -> None:
    """Render quick tips for better usage."""
    with st.expander("💡 Tips for Better Results", expanded=False):
        st.markdown("""
        **Ask specific questions:**
        - Instead of "Tell me about ML", ask "What is the difference between supervised and unsupervised learning?"
        
        **Reference specific topics:**
        - "Explain the backpropagation algorithm from Lecture 3"
        - "What does page 45 say about regularization?"
        
        **Follow-up questions:**
        - The assistant remembers context, so you can ask follow-ups like "Can you give an example?"
        
        **Check sources:**
        - Expand the sources section to verify where the answer came from
        """)


def _render_footer() -> None:
    """Render sidebar footer with links."""
    st.markdown("##### 🔗 Resources")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"[📖 API Docs]({config.API_BASE_URL}/docs)")
    with col2:
        st.markdown(f"[📘 ReDoc]({config.API_BASE_URL}/redoc)")
    
    st.caption("Built with FastAPI + Streamlit + GPT-4o")
    st.caption("© 2026 RAG Teaching Assistant")
