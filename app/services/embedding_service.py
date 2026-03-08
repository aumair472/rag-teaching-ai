"""
Embedding service module.

Wraps sentence-transformers to provide batch encoding,
L2 normalization, and numpy output for downstream FAISS indexing.
"""

from typing import List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Generates text embeddings using sentence-transformers.

    The model is loaded once on initialization and reused for
    all subsequent encode calls.

    Attributes:
        model_name: Name of the sentence-transformers model.
        model: The loaded SentenceTransformer model.
        dimension: Output embedding dimension.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.dimension = settings.embedding_dimension

        logger.info(
            "Loading embedding model",
            extra={"model": self.model_name},
        )
        self.model = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode text(s) into embedding vectors.

        Args:
            texts: A single string or list of strings.
            batch_size: Batch size for encoding.
            normalize: Whether to L2-normalize the output vectors.

        Returns:
            A numpy array of shape ``(n, dimension)`` with dtype float32.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

        embeddings = embeddings.astype(np.float32)

        logger.debug(
            "Texts encoded",
            extra={"count": len(texts), "shape": embeddings.shape},
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string.

        Convenience method that returns a 2D array of shape ``(1, dim)``.

        Args:
            query: The query text.

        Returns:
            Normalized embedding array of shape ``(1, dimension)``.
        """
        return self.encode(query, batch_size=1, normalize=True)
