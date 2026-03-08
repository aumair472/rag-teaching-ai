"""
Dedicated streaming client using httpx.

Handles SSE streaming from the /ask endpoint with proper
timeout handling, cancellation, and error recovery.
"""

from typing import Generator, Optional

import httpx

from ui.config import config


class StreamingClient:
    """
    SSE streaming client for the RAG backend /ask endpoint.

    Uses httpx for streaming HTTP responses. Parses SSE
    ``data:`` lines and yields individual tokens.

    Attributes:
        base_url: Backend API base URL.
        timeout: Streaming timeout in seconds.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.base_url = base_url or config.API_BASE_URL
        self.timeout = timeout or config.STREAM_TIMEOUT

    def stream_response(
        self,
        question: str,
        session_id: str,
    ) -> Generator[str, None, None]:
        """
        Stream response tokens from the /ask endpoint.

        Sends ``stream: true`` to receive SSE events.
        Parses each ``data: <token>`` line and yields it.

        Args:
            question: The user's question.
            session_id: Session identifier.

        Yields:
            Individual tokens as strings. On error, yields
            an error message prefixed with ``⚠️``.
        """
        try:
            with httpx.Client(timeout=httpx.Timeout(self.timeout)) as client:
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
                        if not line:
                            continue

                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                return
                            yield data

        except httpx.TimeoutException:
            yield "\n\n⚠️ *Response timed out. Please try again.*"
        except httpx.ConnectError:
            yield "\n\n⚠️ *Cannot connect to backend. Is it running?*"
        except httpx.HTTPStatusError as exc:
            yield f"\n\n⚠️ *Server error: {exc.response.status_code}*"
        except Exception as exc:
            yield f"\n\n⚠️ *Stream error: {str(exc)[:200]}*"

    def is_available(self) -> bool:
        """Quick check if backend is reachable."""
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False
