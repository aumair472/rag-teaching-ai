# RAG Teaching Assistant — Streamlit UI

A professional chat interface for the RAG Teaching Assistant backend.

## Quick Start

```bash
cd streamlit-ui
pip install -r requirements.txt
streamlit run app.py
```

**Backend must be running** at `http://localhost:8000`.

## Features

- 💬 Chat interface with streaming responses
- 📄 Document ingestion (PDF, PPT, Video)
- 🟢 Live backend health indicator
- 📥 Download conversation as JSON
- 📚 Expandable source citations
- 🎨 Dark mode glassmorphism UI

## Configuration

Edit `.env`:

```
API_BASE_URL=http://localhost:8000
```

## Structure

```
streamlit-ui/
├── app.py                # Main entry point + CSS
├── services/
│   └── api_client.py     # Backend HTTP client
├── components/
│   ├── chat.py           # Chat interface
│   └── sidebar.py        # Sidebar controls
├── .env                  # Config
└── requirements.txt
```
