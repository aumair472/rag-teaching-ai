"""
Speech-to-Text (STT) service module.

Transcribes audio files to text using the OpenAI Whisper API.
Supports WAV, MP3, M4A, WebM, OGG, and FLAC formats.
"""

import io
from typing import Optional

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Max audio file size accepted by the Whisper API (25 MB)
MAX_AUDIO_BYTES = 25 * 1024 * 1024


class STTService:
    """
    Speech-to-Text service powered by OpenAI Whisper.

    Accepts raw audio bytes and returns the transcribed text.

    Attributes:
        client: Async OpenAI client.
        model: Whisper model name (e.g. ``whisper-1``).
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        settings = get_settings()
        self.model = settings.whisper_model
        self.client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)

        logger.info(
            "STTService initialized",
            extra={"model": self.model},
        )

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw bytes of the audio file.
            filename: Original filename (used to hint the MIME type).
            language: Optional BCP-47 language code (e.g. ``"en"``).
                      When ``None`` Whisper auto-detects the language.

        Returns:
            The transcribed text string.

        Raises:
            ValueError: If the audio file exceeds the 25 MB limit.
            RuntimeError: On Whisper API failure.
        """
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise ValueError(
                f"Audio file is too large ({len(audio_bytes) / 1_048_576:.1f} MB). "
                f"Maximum allowed size is 25 MB."
            )

        logger.info(
            "Starting transcription",
            extra={
                "audio_filename": filename,
                "size_kb": round(len(audio_bytes) / 1024, 1),
                "language": language or "auto",
            },
        )

        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename  # OpenAI client reads the .name attribute for MIME

        try:
            kwargs = {
                "model": self.model,
                "file": audio_file,
                "response_format": "text",
            }
            if language:
                kwargs["language"] = language

            transcription: str = await self.client.audio.transcriptions.create(**kwargs)

            logger.info(
                "Transcription complete",
                extra={"chars": len(transcription)},
            )
            return transcription.strip()

        except Exception as exc:
            logger.error(
                "Transcription failed",
                extra={"error": str(exc)},
            )
            raise RuntimeError(f"Whisper transcription failed: {exc}") from exc
