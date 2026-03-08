"""
FastAPI application entry point.

Configures:
    - Application lifespan (startup / shutdown)
    - CORS middleware
    - Logging middleware with request-ID correlation
    - Rate limiting
    - Swagger documentation
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.routes import limiter, router
from app.core.config import get_settings
from app.core.logging import (
    generate_request_id,
    get_logger,
    request_id_ctx,
    setup_logging,
)
from app.db.cache import RedisCache
from app.db.vector_store import FAISSVectorStore
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.memory_service import MemoryService
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService
from app.services.stt_service import STTService
from app.services.tts_service import TTSService

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    On startup:
        - Initialize logging
        - Ensure data/logs directories exist
        - Load embedding model
        - Load FAISS index
        - Initialize Redis cache
        - Wire up all services

    On shutdown:
        - Save FAISS index
        - Log shutdown
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info(
        "Starting RAG Teaching Assistant",
        extra={"version": settings.app_version},
    )

    settings.ensure_directories()

    # ── Initialize components ────────────────────────────────
    embedding_service = EmbeddingService()
    vector_store = FAISSVectorStore()
    cache = RedisCache()
    memory_service = MemoryService(redis_client=cache.client)
    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )
    llm_service = LLMService()
    rag_service = RAGService(
        retrieval_service=retrieval_service,
        llm_service=llm_service,
        memory_service=memory_service,
        cache=cache,
    )

    # ── Voice services ───────────────────────────────────────
    stt_service = STTService()
    tts_service = TTSService()

    # ── Attach to app state ──────────────────────────────────
    app.state.embedding_service = embedding_service
    app.state.vector_store = vector_store
    app.state.cache = cache
    app.state.memory_service = memory_service
    app.state.retrieval_service = retrieval_service
    app.state.llm_service = llm_service
    app.state.rag_service = rag_service
    app.state.stt_service = stt_service
    app.state.tts_service = tts_service

    logger.info("All services initialized and ready")

    yield  # ← Application runs here

    # ── Shutdown ─────────────────────────────────────────────
    try:
        vector_store.save()
        logger.info("FAISS index saved on shutdown")
    except Exception as exc:
        logger.error("Failed to save FAISS index", extra={"error": str(exc)})

    logger.info("RAG Teaching Assistant shutting down")


def create_app() -> FastAPI:
    """
    Application factory.

    Returns:
        A fully configured FastAPI application.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "A production-grade Retrieval-Augmented Generation (RAG) "
            "Teaching Assistant API. Ingest lecture videos, PDFs, and "
            "PowerPoints, then ask questions and receive grounded, "
            "cited answers powered by GPT-4o."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ─────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Rate Limiting ────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Logging Middleware ───────────────────────────────────
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next) -> Response:
        """Inject request ID and log request/response metadata."""
        req_id = request.headers.get("X-Request-ID") or generate_request_id()
        request_id_ctx.set(req_id)

        start = time.time()
        logger.info(
            "Request started",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else "unknown",
            },
        )

        response = await call_next(request)

        latency_ms = (time.time() - start) * 1000
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"

        logger.info(
            "Request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(latency_ms, 2),
            },
        )

        return response

    # ── Routes ───────────────────────────────────────────────
    app.include_router(router)

    return app


# Module-level app instance for uvicorn
app = create_app()
