"""
Pydantic schemas for API request/response models and internal data transfer objects.

All models use strict validation and include OpenAPI documentation metadata.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enumerations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SourceType(str, Enum):
    """Supported ingestion source types."""

    VIDEO = "video"
    PDF = "pdf"
    PPT = "ppt"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ingestion Schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DocumentChunk(BaseModel):
    """A single chunk of text with its metadata."""

    text: str = Field(..., description="The chunk text content")
    source_type: SourceType = Field(..., description="Type of source document")
    source_name: str = Field(..., description="Original file name")
    page: Optional[int] = Field(None, description="Page number (PDF)")
    slide: Optional[int] = Field(None, description="Slide number (PPT)")
    timestamp: Optional[str] = Field(None, description="Timestamp range (video)")
    chunk_index: int = Field(..., description="Index of this chunk within the source")


class IngestRequest(BaseModel):
    """Request body for the ingestion endpoint."""

    file_path: str = Field(..., description="Path to the file to ingest")
    source_type: SourceType = Field(..., description="Type of source")
    source_name: Optional[str] = Field(
        None,
        description="Human-readable name; defaults to filename",
    )


class IngestResponse(BaseModel):
    """Response from the ingestion endpoint."""

    message: str
    chunks_created: int
    source_name: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Query / RAG Schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class Source(BaseModel):
    """A single source reference returned alongside an answer."""

    source_name: str = Field(..., description="Name of the source document")
    source_type: SourceType
    page: Optional[int] = None
    slide: Optional[int] = None
    timestamp: Optional[str] = None
    text_snippet: str = Field(..., description="Relevant text excerpt")
    similarity_score: float = Field(..., description="Vector similarity score")
    rerank_score: Optional[float] = Field(None, description="Cross-encoder re-rank score")


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The student's question",
    )
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Session identifier for conversation memory",
    )
    stream: bool = Field(
        default=False,
        description="If true, response will be streamed via SSE",
    )


class AskResponse(BaseModel):
    """Response from the /ask endpoint."""

    answer: str = Field(..., description="The generated answer")
    sources: List[Source] = Field(
        default_factory=list,
        description="Source documents used",
    )
    cached: bool = Field(default=False, description="Whether the response was cached")
    session_id: str = Field(..., description="Session identifier")
    latency_ms: Optional[float] = Field(
        None,
        description="End-to-end latency in milliseconds",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Voice Schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class VoiceResponse(BaseModel):
    """JSON metadata returned alongside the audio stream from /voice."""

    transcription: str = Field(
        ...,
        description="The text transcribed from the user's audio by Whisper",
    )
    answer: str = Field(..., description="The generated answer text")
    sources: List[Source] = Field(
        default_factory=list,
        description="Source documents used to generate the answer",
    )
    cached: bool = Field(
        default=False,
        description="Whether the text response was served from cache",
    )
    session_id: str = Field(..., description="Session identifier")
    latency_ms: Optional[float] = Field(
        None,
        description="End-to-end pipeline latency in milliseconds",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Health & System Schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Health of individual components",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Evaluation Schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class EvalSample(BaseModel):
    """A single evaluation data point."""

    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


class EvalResult(BaseModel):
    """Aggregated evaluation metrics from RAGAS."""

    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    num_samples: int = 0
