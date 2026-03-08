"""
API client for backend communication.

Handles standard (non-streaming) HTTP requests to the
RAG Teaching Assistant FastAPI backend.
"""

import time
from typing import Any, Dict, Optional

import requests

from ui.config import config


class APIClient:
    """
    Synchronous HTTP client for the RAG backend.

    Handles health checks, standard /ask requests, and /ingest calls.
    All methods return dicts and never raise — errors are captured
    in the return value.

    Attributes:
        base_url: Backend API base URL.
        timeout: Default request timeout in seconds.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: int = 60) -> None:
        self.base_url = base_url or config.API_BASE_URL
        self.timeout = timeout

    def health(self) -> Dict[str, Any]:
        """
        Check backend health status.

        Returns:
            Health dict with status, components, and latency.
        """
        start = time.time()
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            data["latency_ms"] = round((time.time() - start) * 1000, 1)
            return data
        except requests.ConnectionError:
            return {"status": "offline", "error": "Cannot connect to backend"}
        except requests.Timeout:
            return {"status": "offline", "error": "Health check timed out"}
        except Exception as exc:
            return {"status": "offline", "error": str(exc)}

    def ask(self, question: str, session_id: str) -> Dict[str, Any]:
        """
        Send a question (non-streaming).

        Args:
            question: The student's question.
            session_id: Session identifier.

        Returns:
            Response dict with answer, sources, and metadata.
        """
        start = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/ask",
                json={"question": question, "session_id": session_id, "stream": False},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            data["_client_latency_ms"] = round((time.time() - start) * 1000, 1)
            return data
        except requests.Timeout:
            return {"error": "Request timed out. Try again."}
        except requests.ConnectionError:
            return {"error": "Backend unreachable. Is it running?"}
        except requests.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            return {"error": f"API error: {detail}"}
        except Exception as exc:
            return {"error": str(exc)}

    def ingest(
        self,
        file_path: str,
        source_type: str,
        source_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a document.

        Args:
            file_path: Absolute file path on server.
            source_type: pdf | ppt | video.
            source_name: Optional human label.

        Returns:
            Response dict with chunks_created or error.
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
        except requests.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            return {"error": f"Ingestion failed: {detail}"}
        except Exception as exc:
            return {"error": str(exc)}
