"""
Formatting utility functions.

Shared helpers for text formatting, score display, and time conversion.
"""

from typing import Any, Dict, List, Optional


def format_score(score: Optional[float], decimals: int = 3) -> str:
    """Format a float score for display. Returns '—' if None."""
    if score is None:
        return "—"
    return f"{score:.{decimals}f}"


def format_latency(ms: Optional[float]) -> str:
    """Format milliseconds for display with appropriate unit."""
    if ms is None:
        return "—"
    if ms < 1000:
        return f"{ms:.0f}ms"
    elif ms < 60000:
        return f"{ms / 1000:.1f}s"
    else:
        mins = int(ms / 60000)
        secs = (ms % 60000) / 1000
        return f"{mins}m {secs:.0f}s"


def truncate(text: str, max_len: int = 200) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    # Try to truncate at word boundary
    truncated = text[:max_len]
    last_space = truncated.rfind(' ')
    if last_space > max_len * 0.8:
        truncated = truncated[:last_space]
    return truncated.rstrip() + "…"


def build_source_label(source: Dict[str, Any]) -> str:
    """Build a human-readable label for a source reference."""
    name = source.get("source_name", "Unknown")
    stype = source.get("source_type", "")
    
    # Type icons
    type_icons = {
        "pdf": "📕",
        "ppt": "📊",
        "video": "🎥",
    }
    icon = type_icons.get(stype.lower(), "📄")
    
    parts = [f"{icon} {name}"]

    if source.get("page"):
        parts.append(f"p.{source['page']}")
    if source.get("slide"):
        parts.append(f"slide {source['slide']}")
    if source.get("timestamp"):
        parts.append(f"@{source['timestamp']}")

    return " · ".join(parts)


def format_cache_status(cached: bool) -> str:
    """Return a styled cache status string."""
    return "HIT" if cached else "MISS"


def count_sources_by_type(sources: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count sources grouped by type."""
    counts: Dict[str, int] = {}
    for src in sources:
        stype = src.get("source_type", "unknown")
        counts[stype] = counts.get(stype, 0) + 1
    return counts


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def highlight_match(text: str, query: str, max_context: int = 50) -> str:
    """
    Find and highlight a query match in text with surrounding context.
    Returns the snippet with the match marked.
    """
    lower_text = text.lower()
    lower_query = query.lower()
    
    idx = lower_text.find(lower_query)
    if idx == -1:
        return truncate(text, max_context * 2)
    
    start = max(0, idx - max_context)
    end = min(len(text), idx + len(query) + max_context)
    
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    
    return f"{prefix}{text[start:end]}{suffix}"
