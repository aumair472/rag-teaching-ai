"""
Text-to-Speech (TTS) service module.

Converts answer text to audio using the OpenAI TTS API.
Returns raw MP3 bytes ready to stream or play back.
"""

from typing import Literal, Optional

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

TtsVoice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


class TTSService:
    """
    Text-to-Speech service powered by the OpenAI TTS API.

    Converts a text string to MP3 audio bytes.

    Attributes:
        client: Async OpenAI client.
        model: TTS model name (e.g. ``tts-1-hd``).
        voice: Default voice to use.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        settings = get_settings()
        self.model = settings.tts_model
        self.voice: TtsVoice = settings.tts_voice  # type: ignore[assignment]
        self.client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)

        logger.info(
            "TTSService initialized",
            extra={"model": self.model, "voice": self.voice},
        )

    async def synthesize(
        self,
        text: str,
        voice: Optional[TtsVoice] = None,
    ) -> bytes:
        """
        Synthesize text to MP3 audio bytes.

        Args:
            text: The text to convert to speech.
            voice: Override the default voice for this call.

        Returns:
            Raw MP3 bytes of the synthesized speech.

        Raises:
            ValueError: If ``text`` is empty.
            RuntimeError: On TTS API failure.
        """
        text = text.strip()
        if not text:
            raise ValueError("Cannot synthesize empty text.")

        # TTS API has a 4096-character limit per request; truncate with a note.
        if len(text) > 4096:
            logger.warning(
                "Text exceeds 4096 chars — truncating for TTS",
                extra={"original_len": len(text)},
            )
            text = text[:4090] + " ..."

        chosen_voice: TtsVoice = voice or self.voice

        logger.info(
            "Starting TTS synthesis",
            extra={
                "model": self.model,
                "voice": chosen_voice,
                "chars": len(text),
            },
        )

        try:
            response = await self.client.audio.speech.create(
                model=self.model,
                voice=chosen_voice,
                input=text,
                response_format="mp3",
            )

            audio_bytes: bytes = response.content

            logger.info(
                "TTS synthesis complete",
                extra={"bytes": len(audio_bytes)},
            )
            return audio_bytes

        except Exception as exc:
            logger.error(
                "TTS synthesis failed",
                extra={"error": str(exc)},
            )
            raise RuntimeError(f"TTS synthesis failed: {exc}") from exc
