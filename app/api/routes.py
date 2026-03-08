"""
API routes module.

Defines all HTTP endpoints for the RAG Teaching Assistant:
    - POST /ask — Query the teaching assistant
    - POST /ingest — Ingest new documents
    - GET /health — Health check
"""

import time
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings
from app.core.logging import get_logger

from app.models.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SourceType,
    VoiceResponse,
)

logger = get_logger(__name__)

router = APIRouter()

# Rate limiter instance — shared with main.py
limiter = Limiter(key_func=get_remote_address)


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask the teaching assistant a question",
    description=(
        "Submit a question and receive a grounded answer with source citations. "
        "Supports both standard JSON and streaming SSE responses."
    ),
)
@limiter.limit(get_settings().rate_limit)
async def ask(request: Request, body: AskRequest) -> Any:
    """
    Process a student question through the RAG pipeline.

    If ``stream=true``, returns a ``text/event-stream`` response.
    Otherwise, returns a standard JSON ``AskResponse``.
    """
    rag_service = request.app.state.rag_service

    if body.stream:
        async def event_generator():
            async for token in rag_service.ask_stream(
                question=body.question,
                session_id=body.session_id,
            ):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    response = await rag_service.ask(
        question=body.question,
        session_id=body.session_id,
    )
    return response


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest a new document",
    description="Ingest a PDF, PPT, or video file into the vector store.",
)
@limiter.limit("5/minute")
async def ingest(request: Request, body: IngestRequest) -> IngestResponse:
    """
    Ingest a document file into the RAG knowledge base.

    Extracts text, chunks it, generates embeddings, and stores
    in the vector store.
    """
    from pathlib import Path

    from ingestion.chunk import MetadataAwareChunker
    from ingestion.extract_pdf import PDFExtractor
    from ingestion.extract_ppt import PPTExtractor
    from ingestion.transcribe import VideoTranscriber

    embedding_service = request.app.state.embedding_service
    vector_store = request.app.state.vector_store

    file_path = body.file_path
    source_name = body.source_name or Path(file_path).stem

    logger.info(
        "Ingestion started",
        extra={"file_path": file_path, "source_type": body.source_type.value},
    )

    try:
        if body.source_type == SourceType.PDF:
            extractor = PDFExtractor()
            documents = extractor.extract(file_path)
        elif body.source_type == SourceType.PPT:
            extractor = PPTExtractor()
            documents = extractor.extract(file_path)
        elif body.source_type == SourceType.VIDEO:
            logger.info("Starting video transcription", extra={"file_path": file_path})
            transcriber = VideoTranscriber()
            result = transcriber.transcribe(file_path)
            logger.info(
                "Video transcription complete",
                extra={"segments": len(result["segments"]), "source": source_name}
            )
            
            # For video, use segment-aware chunking
            chunker = MetadataAwareChunker()
            chunks = chunker.chunk_video_segments(
                segments=result["segments"],
                source_name=source_name,
            )
            logger.info("Video chunking complete", extra={"chunks": len(chunks)})
            
            # Embed and store
            texts = [c.text for c in chunks]
            logger.info("Generating embeddings for video chunks", extra={"count": len(texts)})
            embeddings = embedding_service.encode(texts)
            metadata = [c.model_dump() for c in chunks]
            vector_store.add(embeddings, metadata)
            vector_store.save()
            logger.info("Video ingestion complete", extra={"source": source_name, "chunks": len(chunks)})

            return IngestResponse(
                message="Video ingested successfully",
                chunks_created=len(chunks),
                source_name=source_name,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported source type: {body.source_type}",
            )

        # PDF / PPT: standard chunking
        chunker = MetadataAwareChunker()
        chunks = chunker.chunk_documents(documents)

        # Embed and store
        texts = [c.text for c in chunks]
        embeddings = embedding_service.encode(texts)
        metadata = [c.model_dump() for c in chunks]
        vector_store.add(embeddings, metadata)
        vector_store.save()

        logger.info(
            "Ingestion complete",
            extra={"source": source_name, "chunks": len(chunks)},
        )

        return IngestResponse(
            message="Document ingested successfully",
            chunks_created=len(chunks),
            source_name=source_name,
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Ingestion failed", extra={"error": str(exc)})
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(exc)}",
        ) from exc


@router.post(
    "/voice",
    summary="Ask a question by voice",
    description=(
        "Upload an audio file (WAV, MP3, WebM, M4A, OGG, FLAC). "
        "The audio is transcribed with Whisper, run through the RAG pipeline, "
        "and the answer is synthesized back as MP3 audio. "
        "The JSON metadata (transcription, answer, sources) is returned in the "
        "X-Voice-Metadata response header as a URL-encoded JSON string."
    ),
    response_class=Response,
    responses={
        200: {
            "content": {"audio/mpeg": {}},
            "description": "MP3 audio of the spoken answer",
        }
    },
)
async def voice(
    request: Request,
    audio: UploadFile = File(..., description="Audio file to transcribe (WAV/MP3/WebM/M4A/OGG/FLAC)"),
    session_id: str = Form(default="default", description="Session identifier for conversation memory"),
) -> Response:
    """
    Voice pipeline: STT → RAG → TTS.

    1. Transcribe the uploaded audio with Whisper.
    2. Pass the transcription through the full RAG pipeline.
    3. Synthesize the answer to MP3 with OpenAI TTS.
    4. Return the MP3 bytes; attach JSON metadata in ``X-Voice-Metadata``.
    """
    import json
    import urllib.parse
    from fastapi.responses import JSONResponse

    try:
        stt_service = request.app.state.stt_service
        tts_service = request.app.state.tts_service
        rag_service = request.app.state.rag_service
    except AttributeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Voice services not initialised: {exc}. Please restart the server.",
        ) from exc

    start = time.time()

    # ── 1. Read audio ────────────────────────────────────────
    try:
        audio_bytes = await audio.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read audio file: {exc}") from exc

    # ── 2. Transcribe ────────────────────────────────────────
    try:
        transcription = await stt_service.transcribe(
            audio_bytes=audio_bytes,
            filename=audio.filename or "audio.wav",
        )
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not transcription:
        raise HTTPException(
            status_code=422,
            detail="Could not transcribe audio. Please speak clearly and try again.",
        )

    logger.info(
        "Voice: transcription ready",
        extra={"session_id": session_id, "transcription": transcription[:120]},
    )

    # ── 3. RAG pipeline ──────────────────────────────────────
    rag_response = await rag_service.ask(
        question=transcription,
        session_id=session_id,
    )

    # ── 4. Synthesize answer to audio ────────────────────────
    try:
        mp3_bytes = await tts_service.synthesize(text=rag_response.answer)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    latency_ms = (time.time() - start) * 1000

    # ── 5. Build metadata header ─────────────────────────────
    metadata = VoiceResponse(
        transcription=transcription,
        answer=rag_response.answer,
        sources=rag_response.sources,
        cached=rag_response.cached,
        session_id=session_id,
        latency_ms=round(latency_ms, 2),
    )
    metadata_header = urllib.parse.quote(
        metadata.model_dump_json(),
        safe="",
    )

    logger.info(
        "Voice: request complete",
        extra={"session_id": session_id, "latency_ms": round(latency_ms, 2)},
    )

    return Response(
        content=mp3_bytes,
        media_type="audio/mpeg",
        headers={
            "X-Voice-Metadata": metadata_header,
            "X-Session-ID": session_id,
        },
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the health status of the application and its components.",
)
async def health(request: Request) -> HealthResponse:
    """Return health status of all system components."""
    settings = get_settings()

    components = {}

    # Vector store
    try:
        vector_store = request.app.state.vector_store
        count = vector_store.count()
        components["vector_store"] = f"healthy ({count} vectors)"
    except Exception:
        components["vector_store"] = "unavailable"

    # Redis
    try:
        cache = request.app.state.cache
        components["redis"] = "healthy" if cache.is_available() else "unavailable"
    except Exception:
        components["redis"] = "unavailable"

    # Embedding model
    try:
        _ = request.app.state.embedding_service
        components["embedding_model"] = "healthy"
    except Exception:
        components["embedding_model"] = "unavailable"

    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        components=components,
    )
