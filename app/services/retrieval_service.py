"""
Hybrid retrieval service module.

Implements a two-stage retrieval pipeline:
    1. Bi-encoder vector similarity search (FAISS)
    2. Cross-encoder re-ranking (cross-encoder/ms-marco-MiniLM-L-6-v2)

Includes similarity-threshold filtering between stages.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.vector_store import VectorStoreBase
from app.services.embedding_service import EmbeddingService

logger = get_logger(__name__)


class RetrievedChunk:
    """
    A retrieved and ranked document chunk.

    Attributes:
        text: The chunk text.
        metadata: Source metadata dict.
        similarity_score: Cosine similarity from bi-encoder.
        rerank_score: Score from cross-encoder re-ranker.
    """

    __slots__ = ("text", "metadata", "similarity_score", "rerank_score")

    def __init__(
        self,
        text: str,
        metadata: Dict[str, Any],
        similarity_score: float,
        rerank_score: float = 0.0,
    ) -> None:
        self.text = text
        self.metadata = metadata
        self.similarity_score = similarity_score
        self.rerank_score = rerank_score

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "text": self.text,
            "metadata": self.metadata,
            "similarity_score": self.similarity_score,
            "rerank_score": self.rerank_score,
        }


class RetrievalService:
    """
    Hybrid retrieval: vector search → threshold filter → cross-encoder rerank.

    The pipeline:
        1. Encode query with bi-encoder
        2. Search FAISS for Top-K candidates
        3. Filter by similarity threshold
        4. Re-rank with cross-encoder
        5. Return Top-N results

    Attributes:
        embedding_service: Bi-encoder for query encoding.
        vector_store: Vector store backend (FAISS).
        cross_encoder: Cross-encoder model for re-ranking.
        top_k: Number of candidates from vector search.
        top_n: Number of results after re-ranking.
        threshold: Minimum similarity score.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreBase,
        cross_encoder_model: Optional[str] = None,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.top_k = top_k or settings.top_k
        self.top_n = top_n or settings.top_n_rerank
        self.threshold = similarity_threshold or settings.similarity_threshold

        model_name = cross_encoder_model or settings.cross_encoder_model
        logger.info(
            "Loading cross-encoder model",
            extra={"model": model_name},
        )
        self.cross_encoder = CrossEncoder(model_name)
        logger.info("Cross-encoder model loaded")

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """
        Execute the full hybrid retrieval pipeline.

        Args:
            query: The user's question.

        Returns:
            A list of ``RetrievedChunk`` objects, sorted by rerank score
            (descending). Empty list if no relevant results found.
        """
        # Step 1: Encode query
        query_vector = self.embedding_service.encode_query(query)

        # Step 2: Vector similarity search
        candidates = self.vector_store.search(query_vector, top_k=self.top_k)

        if not candidates:
            logger.info("No candidates from vector search")
            return []

        logger.info(
            "Vector search results",
            extra={"candidates": len(candidates)},
        )

        # Step 3: Threshold filtering
        filtered: List[Tuple[Dict[str, Any], float]] = [
            (meta, score) for meta, score in candidates if score >= self.threshold
        ]

        if not filtered:
            logger.info(
                "All candidates below threshold",
                extra={"threshold": self.threshold},
            )
            return []

        logger.info(
            "After threshold filtering",
            extra={"remaining": len(filtered), "threshold": self.threshold},
        )

        # Step 4: Cross-encoder re-ranking
        texts = [meta.get("text", "") for meta, _ in filtered]
        pairs = [(query, text) for text in texts]
        rerank_scores = self.cross_encoder.predict(pairs)

        # Build results
        results: List[RetrievedChunk] = []
        for (meta, sim_score), rerank_score in zip(filtered, rerank_scores):
            results.append(
                RetrievedChunk(
                    text=meta.get("text", ""),
                    metadata={
                        k: v
                        for k, v in meta.items()
                        if k != "text"
                    },
                    similarity_score=sim_score,
                    rerank_score=float(rerank_score),
                )
            )

        # Sort by rerank score descending, take top-N
        results.sort(key=lambda r: r.rerank_score, reverse=True)
        results = results[: self.top_n]

        logger.info(
            "Retrieval complete",
            extra={
                "returned": len(results),
                "top_rerank_score": results[0].rerank_score if results else 0.0,
            },
        )

        return results
