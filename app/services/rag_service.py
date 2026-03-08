"""
RAG orchestrator service module.

Coordinates the full RAG pipeline:
    1. Cache lookup
    2. Query embedding
    3. Hybrid retrieval (vector + cross-encoder)
    4. Guardrail check
    5. Prompt construction with conversation history
    6. LLM generation (standard or streaming)
    7. Cache storage
    8. Memory update
"""

import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.cache import RedisCache
from app.models.schemas import AskResponse, Source, SourceType
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.memory_service import MemoryService
from app.services.retrieval_service import RetrievalService, RetrievedChunk

logger = get_logger(__name__)

# ─── Guardrail message ───────────────────────────────────────────────
GUARDRAIL_MESSAGE = (
    "This topic is not covered in the course material. "
    "Please ask a question related to the course content."
)


class RAGService:
    """
    Top-level orchestrator for the RAG Teaching Assistant.

    Coordinates cache, retrieval, guardrails, LLM generation,
    and memory for each query.

    Attributes:
        retrieval_service: Hybrid retrieval service.
        llm_service: LLM generation service.
        memory_service: Conversation memory manager.
        cache: Redis cache for responses.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_service: LLMService,
        memory_service: MemoryService,
        cache: RedisCache,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.cache = cache
        self.settings = get_settings()

        logger.info("RAGService initialized")

    async def ask(
        self,
        question: str,
        session_id: str,
        stream: bool = False,
    ) -> AskResponse:
        """
        Process a student question through the full RAG pipeline.

        Args:
            question: The student's question.
            session_id: Conversation session identifier.
            stream: Whether to stream the response (handled by caller).

        Returns:
            An ``AskResponse`` with the answer and source citations.
        """
        start_time = time.time()

        # ── Step 1: Cache lookup ─────────────────────────────
        cached = self.cache.get(question)
        if cached:
            logger.info("Returning cached response")
            latency = (time.time() - start_time) * 1000
            return AskResponse(
                answer=cached.get("answer", ""),
                sources=[Source(**s) for s in cached.get("sources", [])],
                cached=True,
                session_id=session_id,
                latency_ms=round(latency, 2),
            )

        # ── Step 2: Hybrid retrieval ─────────────────────────
        retrieved_chunks = self.retrieval_service.retrieve(question)

        # ── Step 3: Guardrail check ──────────────────────────
        if not retrieved_chunks:
            logger.info("Guardrail triggered — no relevant chunks")
            latency = (time.time() - start_time) * 1000
            return AskResponse(
                answer=GUARDRAIL_MESSAGE,
                sources=[],
                cached=False,
                session_id=session_id,
                latency_ms=round(latency, 2),
            )

        # ── Step 4: Build context ────────────────────────────
        context_dicts = self._chunks_to_dicts(retrieved_chunks)
        context_str = self.llm_service.format_context(context_dicts)

        # ── Step 5: Get conversation history ─────────────────
        history_str = self.memory_service.format_history(session_id)

        # ── Step 6: LLM generation ───────────────────────────
        answer = await self.llm_service.generate(
            question=question,
            context=context_str,
            history=history_str,
        )

        # ── Step 7: Build sources ────────────────────────────
        sources = self._build_sources(retrieved_chunks)

        # ── Step 8: Cache the response ───────────────────────
        cache_payload = {
            "answer": answer,
            "sources": [s.model_dump() for s in sources],
        }
        self.cache.set(question, cache_payload)

        # ── Step 9: Update memory ────────────────────────────
        self.memory_service.add_turn(session_id, question, answer)

        latency = (time.time() - start_time) * 1000

        logger.info(
            "RAG pipeline complete",
            extra={
                "session_id": session_id,
                "latency_ms": round(latency, 2),
                "sources_count": len(sources),
            },
        )

        return AskResponse(
            answer=answer,
            sources=sources,
            cached=False,
            session_id=session_id,
            latency_ms=round(latency, 2),
        )

    async def ask_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[str, None]:
        """
        Stream the RAG response token-by-token.

        Performs retrieval and guardrail checks synchronously,
        then yields LLM tokens asynchronously for SSE.

        Args:
            question: The student's question.
            session_id: Conversation session identifier.

        Yields:
            Response tokens as strings.
        """
        # Cache check
        cached = self.cache.get(question)
        if cached:
            yield cached.get("answer", "")
            return

        # Retrieval
        retrieved_chunks = self.retrieval_service.retrieve(question)

        if not retrieved_chunks:
            yield GUARDRAIL_MESSAGE
            return

        context_dicts = self._chunks_to_dicts(retrieved_chunks)
        context_str = self.llm_service.format_context(context_dicts)
        history_str = self.memory_service.format_history(session_id)

        # Stream from LLM
        full_answer = ""
        async for token in self.llm_service.generate_stream(
            question=question,
            context=context_str,
            history=history_str,
        ):
            full_answer += token
            yield token

        # Post-stream: cache and memory update
        sources = self._build_sources(retrieved_chunks)
        cache_payload = {
            "answer": full_answer,
            "sources": [s.model_dump() for s in sources],
        }
        self.cache.set(question, cache_payload)
        self.memory_service.add_turn(session_id, question, full_answer)

    @staticmethod
    def _chunks_to_dicts(chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        """Convert RetrievedChunk objects to dicts for context formatting."""
        return [
            {
                "text": c.text,
                "source_name": c.metadata.get("source_name", "Unknown"),
                "source_type": c.metadata.get("source_type", "unknown"),
                "page": c.metadata.get("page"),
                "slide": c.metadata.get("slide"),
                "timestamp": c.metadata.get("timestamp"),
            }
            for c in chunks
        ]

    @staticmethod
    def _build_sources(chunks: List[RetrievedChunk]) -> List[Source]:
        """Build Source schema objects from retrieved chunks."""
        sources: List[Source] = []
        for chunk in chunks:
            source_type_str = chunk.metadata.get("source_type", "pdf")
            try:
                source_type = SourceType(source_type_str)
            except ValueError:
                source_type = SourceType.PDF

            sources.append(
                Source(
                    source_name=chunk.metadata.get("source_name", "Unknown"),
                    source_type=source_type,
                    page=chunk.metadata.get("page"),
                    slide=chunk.metadata.get("slide"),
                    timestamp=chunk.metadata.get("timestamp"),
                    text_snippet=chunk.text[:200],
                    similarity_score=chunk.similarity_score,
                    rerank_score=chunk.rerank_score,
                )
            )
        return sources
