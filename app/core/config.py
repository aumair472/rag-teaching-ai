"""
Application configuration module.

Loads all configuration from environment variables using Pydantic BaseSettings.
Supports .env file loading for local development.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Central configuration for the RAG Teaching Assistant.

    All values are driven by environment variables and can be
    overridden via a .env file placed at the project root.
    """

    # ── OpenAI ──────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI chat model")
    openai_temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # ── Embedding ───────────────────────────────────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model name",
    )
    embedding_dimension: int = Field(default=384)

    # ── Cross-Encoder ───────────────────────────────────────
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for re-ranking",
    )

    # ── Retrieval ───────────────────────────────────────────
    top_k: int = Field(default=10, description="Top-K results from vector search")
    top_n_rerank: int = Field(default=5, description="Top-N results after re-ranking")
    similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity to keep a result",
    )

    # ── FAISS ───────────────────────────────────────────────
    faiss_index_path: str = Field(default="data/faiss_index.bin")
    faiss_metadata_path: str = Field(default="data/metadata.pkl")

    # ── Chunking ────────────────────────────────────────────
    chunk_size: int = Field(default=512, description="Max characters per chunk")
    chunk_overlap: int = Field(default=64, description="Overlap between chunks")

    # ── Redis / Cache ───────────────────────────────────────
    redis_url: str = Field(default="redis://redis:6379/0")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")

    # ── Memory ──────────────────────────────────────────────
    max_conversation_history: int = Field(
        default=10,
        description="Max turns to keep in conversation memory",
    )

    # ── Rate Limiting ───────────────────────────────────────
    rate_limit: str = Field(
        default="20/minute",
        description="Rate limit string for slowapi",
    )

    # ── Voice Agent ──────────────────────────────────────────
    whisper_model: str = Field(
        default="whisper-1",
        description="OpenAI Whisper model for speech-to-text",
    )
    tts_model: str = Field(
        default="tts-1-hd",
        description="OpenAI TTS model for text-to-speech",
    )
    tts_voice: str = Field(
        default="alloy",
        description="TTS voice (alloy, echo, fable, onyx, nova, shimmer)",
    )

    # ── Application ─────────────────────────────────────────
    app_name: str = Field(default="RAG Teaching Assistant")
    app_version: str = Field(default="1.0.0")
    log_level: str = Field(default="INFO")

    # ── Data Directories ────────────────────────────────────
    data_dir: str = Field(default="data")
    logs_dir: str = Field(default="logs")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    def ensure_directories(self) -> None:
        """Create required directories if they do not exist."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """
    Factory function that returns a cached Settings instance.

    Returns:
        Settings: The application settings singleton.
    """
    return Settings()
