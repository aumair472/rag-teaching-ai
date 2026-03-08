"""
RAG Teaching Assistant — Streamlit UI

A production-grade chat interface for the RAG Teaching Assistant
backend. Connects to the FastAPI API at the configured base URL.

Run:
    streamlit run app.py
"""

import streamlit as st

from components.chat import render_chat
from components.sidebar import render_sidebar
from services.api_client import RAGApiClient

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="RAG Teaching Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown(
    """
    <style>
    /* ── Global ─────────────────────────────────────── */
    .stApp {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
    }

    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ── Header ─────────────────────────────────────── */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }

    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }

    .main-header p {
        color: #9ca3af;
        font-size: 0.95rem;
    }

    /* ── Chat messages ──────────────────────────────── */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin-bottom: 0.75rem !important;
        backdrop-filter: blur(10px);
    }

    /* ── Chat input ─────────────────────────────────── */
    .stChatInput > div {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }

    .stChatInput textarea {
        font-size: 0.95rem !important;
    }

    /* ── Sidebar ────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: rgba(15, 15, 35, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    section[data-testid="stSidebar"] .stMarkdown h5 {
        color: #9ca3af;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 0.1em;
    }

    /* ── Buttons ─────────────────────────────────────── */
    .stButton > button {
        border-radius: 8px !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
    }

    /* ── Expander (Sources) ──────────────────────────── */
    .streamlit-expanderHeader {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #9ca3af !important;
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 8px !important;
    }

    /* ── Code blocks ────────────────────────────────── */
    .stCode {
        border-radius: 8px !important;
        font-size: 0.8rem !important;
    }

    /* ── Success / Error alerts ──────────────────────── */
    .stAlert {
        border-radius: 8px !important;
    }

    /* ── Dividers ────────────────────────────────────── */
    hr {
        border-color: rgba(255, 255, 255, 0.06) !important;
    }

    /* ── Scrollbar ───────────────────────────────────── */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.2);
    }

    /* ── Form containers ────────────────────────────── */
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }

    /* ── Download button ────────────────────────────── */
    .stDownloadButton > button {
        background: transparent !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #9ca3af !important;
        font-size: 0.8rem !important;
    }

    /* ── Voice Button ────────────────────────────────── */
    .voice-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 12px rgba(102,126,234,0.3);
    }

    .voice-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 20px rgba(102,126,234,0.5);
    }

    .voice-btn.recording {
        animation: voice-glow 1s ease-in-out infinite;
        background: linear-gradient(135deg, #ef4444, #dc2626);
    }

    @keyframes voice-glow {
        0%, 100% { box-shadow: 0 0 5px rgba(239, 68, 68, 0.5); }
        50% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.8); }
    }

    /* ── Audio player styling ───────────────────────── */
    audio {
        width: 100%;
        border-radius: 8px;
        margin-top: 0.5rem;
    }

    /* ── Tabs (Voice / Text) ────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.02);
        padding: 0.25rem;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-size: 0.9rem;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2)) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    """Entry point for the Streamlit application."""
    # Initialize API client
    client = RAGApiClient()

    # Render sidebar
    render_sidebar(client)

    # Header
    st.markdown(
        """
        <div class="main-header">
            <h1>🎓 RAG Teaching Assistant</h1>
            <p>Ask questions about your course material and get grounded, cited answers.</p>
            <p style="font-size: 0.8rem; color: #667eea; margin-top: 0.5rem;">🎤 Voice Support — speak your question and hear the answer!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Empty state
    if not st.session_state.get("messages"):
        st.markdown(
            """
            <div style="
                text-align: center;
                padding: 3rem 1rem;
                color: #6b7280;
            ">
                <p style="font-size: 2.5rem; margin-bottom: 0.5rem;">💬</p>
                <p style="font-size: 1.1rem; font-weight: 500;">Start a conversation</p>
                <p style="font-size: 0.85rem; max-width: 400px; margin: 0.5rem auto;">
                    Ingest a document first using the sidebar, then ask
                    questions about the course material.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Render chat
    render_chat(client)


if __name__ == "__main__":
    main()
