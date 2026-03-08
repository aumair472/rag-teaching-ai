"""
Video transcription module.

Converts video files to MP3, then transcribes via OpenAI Whisper API.
Outputs structured JSON with timestamps for downstream chunking.
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class VideoTranscriber:
    """
    Transcribes video files using OpenAI's Whisper API.

    Flow:
        1. Extract audio track from video → MP3 via ffmpeg
        2. Send MP3 to Whisper API with ``response_format="verbose_json"``
        3. Return structured segments with timestamps

    Attributes:
        client: An OpenAI client instance.
        settings: Application settings.
    """

    def __init__(self, client: Optional[OpenAI] = None) -> None:
        self.settings = get_settings()
        self.client = client or OpenAI(api_key=self.settings.openai_api_key)

    def _extract_audio(self, video_path: str, output_path: str) -> str:
        """
        Extract audio from a video file using ffmpeg.

        Args:
            video_path: Path to the input video file.
            output_path: Path for the output MP3 file.

        Returns:
            The output MP3 file path.

        Raises:
            RuntimeError: If ffmpeg conversion fails.
        """
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",                 # no video
            "-acodec", "libmp3lame",
            "-ar", "16000",        # 16kHz for Whisper
            "-ac", "1",            # mono
            "-q:a", "4",           # quality
            "-y",                  # overwrite
            output_path,
        ]
        logger.info("Extracting audio from video", extra={"video_path": video_path})
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("ffmpeg failed", extra={"stderr": result.stderr})
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
        logger.info("Audio extraction complete", extra={"output": output_path})
        return output_path

    def transcribe(self, video_path: str) -> Dict[str, Any]:
        """
        Transcribe a video file to structured JSON.

        Args:
            video_path: Path to the video file.

        Returns:
            Dictionary containing:
                - ``source_name``: name of the video file
                - ``source_type``: ``"video"``
                - ``segments``: list of dicts with ``start``, ``end``, ``text``
                - ``full_text``: concatenated transcript
        """
        video_path = str(Path(video_path).resolve())
        source_name = Path(video_path).stem

        with tempfile.TemporaryDirectory() as tmp_dir:
            mp3_path = os.path.join(tmp_dir, f"{source_name}.mp3")
            self._extract_audio(video_path, mp3_path)

            logger.info("Sending audio to Whisper API", extra={"source": source_name})
            with open(mp3_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )

        segments: List[Dict[str, Any]] = []
        for seg in response.segments:  # type: ignore[attr-defined]
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })

        full_text = " ".join(s["text"] for s in segments)

        result = {
            "source_name": source_name,
            "source_type": "video",
            "segments": segments,
            "full_text": full_text,
        }

        logger.info(
            "Transcription complete",
            extra={"source": source_name, "num_segments": len(segments)},
        )
        return result
