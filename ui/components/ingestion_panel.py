"""
Document ingestion panel component.

Professional ingestion form with file path input,
source type selection, progress indication, and notifications.
"""

from typing import Any, Dict, Optional

import streamlit as st

from ui.config import config
from ui.services.api_client import APIClient


def render_ingestion_panel(api_client: APIClient) -> None:
    """
    Render the document ingestion panel.

    Provides a form for ingesting PDF, PPT, or video files
    into the RAG knowledge base.

    Args:
        api_client: The API client for backend communication.
    """
    st.markdown("##### 📄 Ingest Document")

    with st.form("ingestion_form", clear_on_submit=True):
        file_path = st.text_input(
            "File Path",
            placeholder="/absolute/path/to/document.pdf",
            help="Full path to the file on the server filesystem",
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            source_type = st.selectbox(
                "Type",
                options=config.DEFAULT_SOURCE_TYPES,
                format_func=lambda x: {"pdf": "📕 PDF", "ppt": "📊 PPT", "video": "🎥 Video"}.get(x, x),
            )

        with col2:
            source_name = st.text_input(
                "Label (optional)",
                placeholder="e.g. Lecture 3",
                help="Human-readable name for citation",
            )

        submitted = st.form_submit_button(
            "📤 Ingest Document",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        _handle_ingestion(api_client, file_path, source_type, source_name)


def _handle_ingestion(
    api_client: APIClient,
    file_path: str,
    source_type: str,
    source_name: str,
) -> None:
    """
    Execute the ingestion and display results.

    Args:
        api_client: API client.
        file_path: Path to ingest.
        source_type: pdf | ppt | video.
        source_name: Optional label.
    """
    if not file_path or not file_path.strip():
        st.error("⚠️ Please enter a file path.")
        return

    progress_bar = st.progress(0, text="Preparing ingestion…")

    try:
        progress_bar.progress(20, text="Sending to backend…")

        result = api_client.ingest(
            file_path=file_path.strip(),
            source_type=source_type,
            source_name=source_name.strip() or None,
        )

        progress_bar.progress(90, text="Processing response…")

        if "error" in result:
            progress_bar.empty()
            st.error(f"❌ {result['error']}")
        else:
            progress_bar.progress(100, text="Complete!")
            chunks = result.get("chunks_created", 0)
            name = result.get("source_name", "Document")
            st.success(f"✅ **{name}** ingested — **{chunks}** chunks created")

            # Log
            with st.expander("📋 Ingestion Details", expanded=False):
                st.json(result)

    except Exception as exc:
        progress_bar.empty()
        st.error(f"❌ Unexpected error: {str(exc)[:200]}")
