"""
Vector store abstraction and FAISS implementation.

Provides an abstract base class ``VectorStoreBase`` so the backend
can be swapped to Pinecone, Qdrant, or Weaviate without changing
retrieval logic.
"""

import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class VectorStoreBase(ABC):
    """
    Abstract interface for vector stores.

    Any concrete implementation must support:
        - Adding vectors with metadata
        - Searching by vector similarity
        - Persistence (save / load)
        - Deletion
    """

    @abstractmethod
    def add(
        self,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]],
    ) -> None:
        """Add vectors with associated metadata."""
        ...

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for nearest neighbors.

        Returns:
            List of (metadata_dict, distance) tuples.
        """
        ...

    @abstractmethod
    def save(self) -> None:
        """Persist the index and metadata to disk."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load the index and metadata from disk."""
        ...

    @abstractmethod
    def delete(self) -> None:
        """Delete the index and metadata from disk."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the number of vectors in the index."""
        ...


class FAISSVectorStore(VectorStoreBase):
    """
    FAISS-backed vector store with normalized inner-product search.

    Uses ``IndexFlatIP`` (inner product on L2-normalized vectors),
    which is equivalent to cosine similarity.

    Metadata is stored in a companion pandas DataFrame, persisted
    alongside the FAISS index via pickle.

    Attributes:
        dimension: Embedding dimension.
        index: The FAISS index.
        metadata_df: DataFrame holding metadata for each vector.
        index_path: File path for the serialized FAISS index.
        metadata_path: File path for the serialized metadata.
    """

    def __init__(
        self,
        dimension: Optional[int] = None,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.dimension = dimension or settings.embedding_dimension
        self.index_path = index_path or settings.faiss_index_path
        self.metadata_path = metadata_path or settings.faiss_metadata_path
        self.index: Optional[faiss.Index] = None
        self.metadata_df: pd.DataFrame = pd.DataFrame()

        self._init_index()

    def _init_index(self) -> None:
        """Initialize a new FAISS index or load an existing one."""
        if Path(self.index_path).exists() and Path(self.metadata_path).exists():
            self.load()
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata_df = pd.DataFrame()
            logger.info(
                "Initialized new FAISS index",
                extra={"dimension": self.dimension},
            )

    def add(
        self,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]],
    ) -> None:
        """
        Add vectors and their metadata to the store.

        Vectors are L2-normalized before insertion to ensure cosine
        similarity via inner product.

        Args:
            vectors: Shape ``(n, dimension)`` float32 array.
            metadata: List of metadata dictionaries, one per vector.

        Raises:
            ValueError: If vectors and metadata counts don't match.
        """
        if len(vectors) != len(metadata):
            raise ValueError(
                f"Vector count ({len(vectors)}) != metadata count ({len(metadata)})"
            )

        # Normalize for cosine similarity via IP
        faiss.normalize_L2(vectors)

        self.index.add(vectors.astype(np.float32))  # type: ignore[union-attr]

        new_meta = pd.DataFrame(metadata)
        self.metadata_df = pd.concat(
            [self.metadata_df, new_meta], ignore_index=True
        )

        logger.info(
            "Vectors added to FAISS",
            extra={"added": len(vectors), "total": self.count()},
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for the top-K most similar vectors.

        Args:
            query_vector: Shape ``(1, dimension)`` or ``(dimension,)`` float32.
            top_k: Number of results to return.

        Returns:
            List of ``(metadata_dict, similarity_score)`` tuples,
            sorted by descending similarity.
        """
        if self.index is None or self.count() == 0:
            logger.warning("Search called on empty index")
            return []

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Normalize query
        faiss.normalize_L2(query_vector)

        top_k = min(top_k, self.count())
        scores, indices = self.index.search(query_vector.astype(np.float32), top_k)

        results: List[Tuple[Dict[str, Any], float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata_df.iloc[idx].to_dict()
            results.append((meta, float(score)))

        return results

    def save(self) -> None:
        """Persist FAISS index and metadata DataFrame to disk."""
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.metadata_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, self.index_path)

        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata_df, f)

        logger.info(
            "FAISS index saved",
            extra={
                "index_path": self.index_path,
                "metadata_path": self.metadata_path,
                "total_vectors": self.count(),
            },
        )

    def load(self) -> None:
        """Load FAISS index and metadata DataFrame from disk."""
        if not Path(self.index_path).exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not Path(self.metadata_path).exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

        self.index = faiss.read_index(self.index_path)

        with open(self.metadata_path, "rb") as f:
            self.metadata_df = pickle.load(f)

        logger.info(
            "FAISS index loaded",
            extra={"total_vectors": self.count()},
        )

    def delete(self) -> None:
        """Remove index and metadata files from disk and reset in-memory state."""
        for path in (self.index_path, self.metadata_path):
            if Path(path).exists():
                os.remove(path)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_df = pd.DataFrame()
        logger.info("FAISS index deleted and reset")

    def count(self) -> int:
        """Return the number of vectors currently in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal
