"""
API client for the RAG Teaching Assistant backend.

Handles all HTTP communication with the FastAPI backend,
including standard requests, streaming SSE responses,
and error handling with graceful fallbacks.
"""

import json
import os
import urllib.parse
from typing import Any, Dict, Generator, Optional, Tuple

import httpx
import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
DEFAULT_TIMEOUT: int = 60


class RAGApiClient:
    """
    Client for the RAG Teaching Assistant API.

    Provides methods for health checks, question answering
    (standard + streaming), and document ingestion.

    Attributes:
        base_url: The backend API base URL.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = (base_url or API_BASE_URL).rstrip("/")
        self.timeout = timeout

    # ─── Health ─────────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        """
        Check backend health.

        Returns:
            Health response dict, or error dict on failure.
        """
        try:
            resp = requests.get(
                f"{self.base_url}/health",
                timeout=5,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            return {"status": "offline", "error": str(exc)}

    # ─── Ask (standard) ────────────────────────────────────

    def ask(
        self,
        question: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Send a question to the /ask endpoint (non-streaming).

        Args:
            question: The student's question.
            session_id: Session identifier.

        Returns:
            The API response dict with answer and sources.
        """
        try:
            resp = requests.post(
                f"{self.base_url}/ask",
                json={
                    "question": question,
                    "session_id": session_id,
                    "stream": False,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.Timeout:
            return {"error": "Request timed out. Please try again."}
        except requests.ConnectionError:
            return {"error": "Cannot connect to the backend. Is it running?"}
        except Exception as exc:
            return {"error": f"Request failed: {str(exc)}"}

    # ─── Ask (streaming) ───────────────────────────────────

    def ask_stream(
        self,
        question: str,
        session_id: str,
    ) -> Generator[str, None, None]:
        """
        Send a question and stream the response token-by-token.

        Uses httpx for streaming SSE support.

        Args:
            question: The student's question.
            session_id: Session identifier.

        Yields:
            Individual response tokens as strings.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/ask",
                    json={
                        "question": question,
                        "session_id": session_id,
                        "stream": True,
                    },
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            yield data
        except httpx.TimeoutException:
            yield "\n\n⚠️ *Stream timed out. Please try again.*"
        except httpx.ConnectError:
            yield "\n\n⚠️ *Cannot connect to the backend.*"
        except Exception as exc:
            yield f"\n\n⚠️ *Streaming error: {str(exc)}*"

    # ─── Ingest ─────────────────────────────────────────────

    def ingest(
        self,
        file_path: str,
        source_type: str,
        source_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a document into the RAG knowledge base.

        Args:
            file_path: Path to the file on the server.
            source_type: One of "pdf", "ppt", "video".
            source_name: Optional human-readable name.

        Returns:
            The API response dict.
        """
        payload: Dict[str, Any] = {
            "file_path": file_path,
            "source_type": source_type,
        }
        if source_name:
            payload["source_name"] = source_name

        # Use extended timeout for ingestion (especially for videos)
        # Video processing can take several minutes (ffmpeg + Whisper API + embeddings)
        ingestion_timeout = 600  # 10 minutes

        try:
            resp = requests.post(
                f"{self.base_url}/ingest",
                json=payload,
                timeout=ingestion_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.Timeout:
            return {"error": "Ingestion timed out. Large videos may take several minutes to process."}
        except requests.ConnectionError:
            return {"error": "Cannot connect to the backend."}
        except requests.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            return {"error": f"Ingestion failed: {detail}"}
        except Exception as exc:
            return {"error": f"Ingestion failed: {str(exc)}"}

    # ─── Voice ──────────────────────────────────────────────

    def voice_ask(
        self,
        audio_bytes: bytes,
        filename: str,
        session_id: str,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Send audio to ``POST /voice`` and return (mp3_bytes, metadata_dict).

        The server transcribes the audio with Whisper, runs the RAG pipeline,
        synthesizes the answer with TTS, and returns the MP3 audio.
        JSON metadata (transcription, answer, sources) is decoded from the
        ``X-Voice-Metadata`` response header.

        Args:
            audio_bytes: Raw bytes of the audio recording.
            filename: Original filename (e.g. ``"recording.wav"``).
            session_id: Session identifier for conversation memory.

        Returns:
            A tuple of ``(mp3_bytes, metadata_dict)``.
            On error, ``mp3_bytes`` is ``b""`` and ``metadata_dict``
            contains an ``"error"`` key.
        """
        try:
            resp = requests.post(
                f"{self.base_url}/voice",
                files={"audio": (filename, audio_bytes)},
                data={"session_id": session_id},
                timeout=self.timeout,
            )
            resp.raise_for_status()

            # Decode JSON metadata from response header
            raw_header = resp.headers.get("X-Voice-Metadata", "{}")
            metadata: Dict[str, Any] = json.loads(urllib.parse.unquote(raw_header))

            return resp.content, metadata

        except requests.Timeout:
            return b"", {"error": "Voice request timed out. Please try again."}
        except requests.ConnectionError:
            return b"", {"error": "Cannot connect to the backend. Is it running?"}
        except requests.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            return b"", {"error": f"Voice request failed: {detail}"}
        except Exception as exc:
            return b"", {"error": f"Voice request failed: {str(exc)}"}
