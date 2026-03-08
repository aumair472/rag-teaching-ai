"""
Metadata-aware text chunking module.

Splits extracted text into overlapping chunks while preserving
source metadata (type, name, page/slide/timestamp).
Uses sentence-boundary detection for more natural splits.
"""

import re
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import DocumentChunk, SourceType

logger = get_logger(__name__)

# Sentence-ending pattern for boundary detection
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


class MetadataAwareChunker:
    """
    Chunks text into overlapping segments while preserving metadata.

    Strategy:
        1. Split on paragraph boundaries first.
        2. If a paragraph exceeds ``chunk_size``, split on sentence boundaries.
        3. As a last resort, split at ``chunk_size`` character boundaries.
        4. Apply ``chunk_overlap`` to create overlapping windows.

    Attributes:
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks respecting paragraph and sentence boundaries.

        Args:
            text: The raw text to split.

        Returns:
            A list of text chunks.
        """
        # Step 1: split on paragraph breaks
        paragraphs = _PARAGRAPH_BREAK.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        raw_chunks: List[str] = []
        for para in paragraphs:
            if len(para) <= self.chunk_size:
                raw_chunks.append(para)
            else:
                # Step 2: split on sentence boundaries
                sentences = _SENTENCE_END.split(para)
                raw_chunks.extend(self._merge_sentences(sentences))

        # Step 3: apply overlap
        return self._apply_overlap(raw_chunks)

    def _merge_sentences(self, sentences: List[str]) -> List[str]:
        """
        Merge sentences into chunks that fit within chunk_size.

        Args:
            sentences: Individual sentences.

        Returns:
            Merged chunks.
        """
        chunks: List[str] = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current) + len(sentence) + 1 <= self.chunk_size:
                current = f"{current} {sentence}".strip()
            else:
                if current:
                    chunks.append(current)
                # Handle individual sentences longer than chunk_size
                if len(sentence) > self.chunk_size:
                    for i in range(0, len(sentence), self.chunk_size):
                        chunks.append(sentence[i : i + self.chunk_size])
                    current = ""
                else:
                    current = sentence

        if current:
            chunks.append(current)

        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Create overlapping chunks by prepending tail of previous chunk.

        Args:
            chunks: Non-overlapping chunks.

        Returns:
            Chunks with overlap applied.
        """
        if not chunks or self.chunk_overlap == 0:
            return chunks

        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-self.chunk_overlap :]
            overlapped.append(f"{prev_tail} {chunks[i]}".strip())

        return overlapped

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[DocumentChunk]:
        """
        Chunk a list of extracted documents.

        Args:
            documents: List of dicts from PDF/PPT/video extractors.
                Each must contain ``text``, ``source_type``, ``source_name``.
                Optionally ``page``, ``slide``, or ``timestamp``.

        Returns:
            A list of ``DocumentChunk`` objects.
        """
        all_chunks: List[DocumentChunk] = []
        chunk_index = 0

        for doc in documents:
            text = doc.get("text", "")
            if not text.strip():
                continue

            splits = self._split_text(text)

            for split_text in splits:
                chunk = DocumentChunk(
                    text=split_text,
                    source_type=SourceType(doc["source_type"]),
                    source_name=doc["source_name"],
                    page=doc.get("page"),
                    slide=doc.get("slide"),
                    timestamp=doc.get("timestamp"),
                    chunk_index=chunk_index,
                )
                all_chunks.append(chunk)
                chunk_index += 1

        logger.info(
            "Chunking complete",
            extra={"total_chunks": len(all_chunks)},
        )
        return all_chunks

    def chunk_video_segments(
        self,
        segments: List[Dict[str, Any]],
        source_name: str,
    ) -> List[DocumentChunk]:
        """
        Chunk video transcript segments with timestamp metadata.

        Merges short segments until they reach chunk_size, preserving
        first-segment start and last-segment end timestamps.

        Args:
            segments: Whisper transcript segments with ``start``, ``end``, ``text``.
            source_name: Name of the video source.

        Returns:
            A list of ``DocumentChunk`` objects.
        """
        all_chunks: List[DocumentChunk] = []
        chunk_index = 0
        current_text = ""
        start_time: Optional[float] = None
        end_time: Optional[float] = None

        for seg in segments:
            seg_text = seg.get("text", "").strip()
            if not seg_text:
                continue

            if start_time is None:
                start_time = seg["start"]

            if len(current_text) + len(seg_text) + 1 <= self.chunk_size:
                current_text = f"{current_text} {seg_text}".strip()
                end_time = seg["end"]
            else:
                # Emit current chunk
                if current_text:
                    timestamp_str = f"{self._fmt_time(start_time)}-{self._fmt_time(end_time)}"
                    all_chunks.append(
                        DocumentChunk(
                            text=current_text,
                            source_type=SourceType.VIDEO,
                            source_name=source_name,
                            timestamp=timestamp_str,
                            chunk_index=chunk_index,
                        )
                    )
                    chunk_index += 1

                current_text = seg_text
                start_time = seg["start"]
                end_time = seg["end"]

        # Emit last chunk
        if current_text:
            timestamp_str = f"{self._fmt_time(start_time)}-{self._fmt_time(end_time)}"
            all_chunks.append(
                DocumentChunk(
                    text=current_text,
                    source_type=SourceType.VIDEO,
                    source_name=source_name,
                    timestamp=timestamp_str,
                    chunk_index=chunk_index,
                )
            )

        logger.info(
            "Video chunking complete",
            extra={"source": source_name, "total_chunks": len(all_chunks)},
        )
        return all_chunks

    @staticmethod
    def _fmt_time(seconds: Optional[float]) -> str:
        """Format seconds to MM:SS string."""
        if seconds is None:
            return "00:00"
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
