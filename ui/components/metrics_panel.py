"""
Metrics panel component.

Displays RAG pipeline transparency metrics: latency, cache status,
source counts, and retrieval scores. Enhanced visualization.
"""

from typing import Any, Dict, List, Optional

import streamlit as st

from ui.state.session_manager import SessionManager
from ui.utils.formatting import format_cache_status, format_latency, format_score


def render_metrics_panel() -> None:
    """
    Render the RAG transparency panel.

    Shows the latest response metrics including latency,
    cache status, and source breakdown.
    """
    metrics = SessionManager.get_metrics()

    if not metrics:
        st.markdown(
            """<div style="text-align: center; padding: 1rem; color: #6b7280;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📈</div>
                <div style="font-size: 0.85rem;">No metrics yet</div>
                <div style="font-size: 0.75rem; margin-top: 0.25rem;">Ask a question to see RAG pipeline details</div>
            </div>""",
            unsafe_allow_html=True
        )
        return

    # ── Latency + Cache Row ─────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        latency = metrics.get("latency_ms")
        if latency:
            cls = "good" if latency < 2000 else "warn" if latency < 5000 else "bad"
            icon = "⚡" if latency < 2000 else "⏱️" if latency < 5000 else "🐢"
        else:
            cls = ""
            icon = "⏱️"
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Latency</div>
                <div class="value {cls}">{icon} {format_latency(latency)}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col2:
        cached = metrics.get("cached", False)
        cache_icon = "⚡" if cached else "🔄"
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Cache</div>
                <div class="value {'good' if cached else ''}">{cache_icon} {format_cache_status(cached)}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col3:
        count = metrics.get("sources_count", 0)
        src_icon = "📚" if count > 0 else "📭"
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Sources</div>
                <div class="value">{src_icon} {count}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Performance indicator ───────────────────────────────
    latency = metrics.get("latency_ms", 0)
    if latency:
        if latency < 2000:
            st.success("✅ Excellent response time!", icon="⚡")
        elif latency < 5000:
            st.info("Good response time", icon="👍")
        else:
            st.warning("Response was slow. Try a simpler question.", icon="⏳")

    # ── Client Latency ──────────────────────────────────────
    client_lat = metrics.get("client_latency_ms")
    if client_lat:
        st.caption(f"🌐 Network round-trip: {format_latency(client_lat)}")


def render_inline_metrics(
    latency_ms: Optional[float] = None,
    cached: bool = False,
    sources_count: int = 0,
) -> None:
    """
    Render a compact inline metrics bar.

    Used directly below a response in the chat window.

    Args:
        latency_ms: Response latency.
        cached: Whether response was cached.
        sources_count: Number of sources.
    """
    parts = []
    if cached:
        parts.append("⚡ Cached")
    if latency_ms:
        parts.append(f"⏱️ {format_latency(latency_ms)}")
    if sources_count:
        parts.append(f"📚 {sources_count} sources")

    if parts:
        st.caption(" · ".join(parts))
