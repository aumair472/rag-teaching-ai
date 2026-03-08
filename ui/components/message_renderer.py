"""
Message renderer component.

Renders individual chat messages with markdown, source citations,
and metadata badges. Enhanced UX with visual indicators.
"""

from typing import Any, Dict, List

import streamlit as st

from ui.utils.formatting import (
    build_source_label,
    format_cache_status,
    format_latency,
    format_score,
    truncate,
)


def render_message(msg: Dict[str, Any]) -> None:
    """
    Render a single message from conversation history.

    Args:
        msg: Message dict with role, content, and optional metadata.
    """
    role = msg["role"]
    avatar = "🧑‍🎓" if role == "user" else "🤖"

    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

        if role == "assistant":
            _render_message_footer(msg)


def _render_message_footer(msg: Dict[str, Any]) -> None:
    """Render metadata badges below an assistant message."""
    # Build footer badges
    footer_html = '<div class="response-footer">'
    
    if msg.get("cached"):
        footer_html += '<span class="footer-badge cached">⚡ Cached</span>'
    
    if msg.get("latency_ms"):
        latency = msg["latency_ms"]
        speed_class = "fast" if latency < 2000 else "slow" if latency > 5000 else ""
        footer_html += f'<span class="footer-badge {speed_class}">⏱️ {format_latency(latency)}</span>'
    
    sources = msg.get("sources", [])
    if sources:
        footer_html += f'<span class="footer-badge">📚 {len(sources)} sources</span>'
    
    footer_html += '</div>'
    
    if msg.get("cached") or msg.get("latency_ms") or sources:
        st.markdown(footer_html, unsafe_allow_html=True)

    # Sources expander
    if sources:
        render_sources(sources)


def render_sources(sources: List[Dict[str, Any]]) -> None:
    """
    Render source citations in a beautiful expandable card layout.

    Args:
        sources: List of source dicts from the API.
    """
    with st.expander(f"📚 View Sources ({len(sources)})", expanded=False):
        # View toggle
        view_mode = st.radio(
            "View",
            ["List", "Compact"],
            horizontal=True,
            label_visibility="collapsed",
            key=f"source_view_{id(sources)}"
        )
        
        if view_mode == "Compact":
            _render_compact_sources(sources)
        else:
            _render_detailed_sources(sources)


def _render_detailed_sources(sources: List[Dict[str, Any]]) -> None:
    """Render sources in detailed list view."""
    for i, src in enumerate(sources):
        label = build_source_label(src)
        snippet = truncate(src.get("text_snippet", ""), 300)
        sim_score = src.get("similarity_score", 0)
        rerank_score = src.get("rerank_score", 0)
        source_type = src.get("source_type", "pdf").upper()
        
        # Determine score color class
        score_class = "high" if sim_score > 0.6 else "medium" if sim_score > 0.4 else "low"
        
        st.markdown(
            f"""<div class="source-card">
                <div class="source-header">
                    <div class="source-title">
                        [{i+1}] {label}
                        <span class="source-type-badge">{source_type}</span>
                    </div>
                </div>
                <div class="source-meta">
                    <span class="score {score_class}">📊 Similarity: {format_score(sim_score)}</span>
                    <span class="score">🎯 Re-rank: {format_score(rerank_score)}</span>
                </div>
                <div class="source-snippet">{snippet}</div>
            </div>""",
            unsafe_allow_html=True,
        )


def _render_compact_sources(sources: List[Dict[str, Any]]) -> None:
    """Render sources in compact grid view."""
    st.markdown('<div class="sources-grid">', unsafe_allow_html=True)
    
    cols = st.columns(2)
    for i, src in enumerate(sources):
        with cols[i % 2]:
            label = build_source_label(src)
            sim_score = src.get("similarity_score", 0)
            score_class = "high" if sim_score > 0.6 else "medium" if sim_score > 0.4 else "low"
            
            st.markdown(
                f"""<div class="source-card" style="margin-bottom: 8px;">
                    <div class="source-title">[{i+1}] {label}</div>
                    <div class="source-meta">
                        <span class="score {score_class}">📊 {format_score(sim_score)}</span>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
    
    st.markdown('</div>', unsafe_allow_html=True)
