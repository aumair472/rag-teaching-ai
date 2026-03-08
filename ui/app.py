"""
RAG Teaching Assistant — Advanced Streamlit UI

Enterprise-grade chat interface for the RAG Teaching Assistant.
Connects to the FastAPI backend and provides:
    - Streaming chat with source citations
    - RAG pipeline transparency metrics
    - Document ingestion with progress tracking
    - Session management with import/export
    - Live backend health monitoring

Run:
    streamlit run ui/app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure the project root is on sys.path so `ui.*` imports resolve
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ui.components.chat_window import render_chat_window
from ui.components.sidebar import render_sidebar
from ui.config import config
from ui.services.api_client import APIClient
from ui.services.streaming_client import StreamingClient
from ui.state.session_manager import SessionManager

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Suggested Questions for New Users
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUGGESTED_QUESTIONS = [
    {"icon": "🧠", "text": "What is gradient descent?", "category": "Fundamentals"},
    {"icon": "🔄", "text": "Explain backpropagation step by step", "category": "Neural Networks"},
    {"icon": "📊", "text": "What are the types of machine learning?", "category": "Overview"},
    {"icon": "⚡", "text": "How does regularization prevent overfitting?", "category": "Optimization"},
    {"icon": "🎯", "text": "What is cross-validation and why use it?", "category": "Evaluation"},
    {"icon": "🌳", "text": "Explain decision trees vs random forests", "category": "Models"},
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": f"{config.API_BASE_URL}/docs",
        "Report a bug": None,
        "About": (
            "**RAG Teaching Assistant**\n\n"
            "Production-grade Retrieval-Augmented Generation system "
            "for AI-powered course Q&A.\n\n"
            "**Features:**\n"
            "- Hybrid retrieval (semantic + re-ranking)\n"
            "- Source citations with page numbers\n"
            "- Voice input/output support\n"
            "- Conversation memory"
        ),
    },
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Inject Custom CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_css_path = Path(__file__).parent / "styles" / "custom.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    """Application entry point."""

    # Initialize session state
    SessionManager.initialize()

    # Create clients
    api_client = APIClient()
    streaming_client = StreamingClient()

    # ── Sidebar ─────────────────────────────────────────────
    render_sidebar(api_client)

    # ── Hero Header ─────────────────────────────────────────
    st.markdown(
        """<div class="hero-card">
            <h1>🎓 RAG Teaching Assistant</h1>
            <p class="subtitle">
                Ask questions about your course material — get grounded,
                cited answers powered by hybrid retrieval + GPT-4o.
            </p>
            <div class="hero-badges">
                <span class="hero-badge">🔍 Semantic Search</span>
                <span class="hero-badge">📄 Source Citations</span>
                <span class="hero-badge">🎤 Voice Support</span>
                <span class="hero-badge">💬 Memory</span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Empty State with Suggested Questions ────────────────
    if SessionManager.message_count() == 0:
        st.markdown(
            """<div class="empty-state">
                <div class="icon">💬</div>
                <div class="title">Start a conversation</div>
                <div class="desc">
                    Ingest a document using the sidebar, then ask
                    questions about the course material. Responses are
                    grounded in your documents with full source citations.
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        
        # Suggested Questions
        st.markdown("#### 💡 Try asking...")
        
        cols = st.columns(3)
        for idx, q in enumerate(SUGGESTED_QUESTIONS):
            with cols[idx % 3]:
                if st.button(
                    f"{q['icon']} {q['text'][:40]}{'...' if len(q['text']) > 40 else ''}",
                    key=f"suggested_{idx}",
                    use_container_width=True,
                    help=f"Category: {q['category']}"
                ):
                    # Store the question to be processed
                    st.session_state["pending_question"] = q["text"]
                    st.rerun()
        
        st.markdown("---")

    # ── Chat Window ─────────────────────────────────────────
    render_chat_window(api_client, streaming_client)


if __name__ == "__main__":
    main()
