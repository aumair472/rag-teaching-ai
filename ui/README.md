# RAG Teaching Assistant — Advanced UI

Enterprise-grade Streamlit frontend for the RAG Teaching Assistant backend.

## Quick Start

```bash
# From the project root (rag-assistant/)
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

Opens at **http://localhost:8501**. Backend must be running at **http://localhost:8000**.

## Architecture

```
ui/
├── app.py                         # Entry point + CSS injection
├── config.py                      # .env-driven configuration
├── state/
│   └── session_manager.py         # st.session_state abstraction
├── services/
│   ├── api_client.py              # Standard HTTP client
│   └── streaming_client.py        # httpx SSE streaming client
├── components/
│   ├── chat_window.py             # Main chat with streaming
│   ├── message_renderer.py        # Message + source rendering
│   ├── sidebar.py                 # Full sidebar orchestrator
│   ├── metrics_panel.py           # RAG transparency metrics
│   └── ingestion_panel.py         # Document ingestion form
├── styles/
│   └── custom.css                 # Enterprise CSS theme
└── utils/
    └── formatting.py              # Shared formatting helpers
```

## Features

- 💬 Streaming chat (SSE token-by-token)
- 📚 Expandable source citations with score display
- 📊 RAG pipeline transparency panel
- 📄 Document ingestion with progress bar
- 🟢 Live backend health monitor
- 📥 Export / 📤 Import conversations
- 🎨 Enterprise dark glassmorphism theme
- ⚡ Cache hit/miss indicators
- ⏱️ Latency metrics

## Configuration

Edit `ui/.env`:
```
API_BASE_URL=http://localhost:8000
STREAM_TIMEOUT=60
```
