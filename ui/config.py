"""
UI configuration module.

Loads all UI settings from environment variables via python-dotenv.
No hardcoded URLs or secrets.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from the ui/ directory
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)


class UIConfig:
    """
    Centralized UI configuration.

    All values are read from environment variables at import time.

    Attributes:
        API_BASE_URL: Backend API base URL.
        STREAM_TIMEOUT: Timeout for streaming requests in seconds.
        HEALTH_CHECK_INTERVAL: Seconds between health check refreshes.
    """

    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
    STREAM_TIMEOUT: int = int(os.getenv("STREAM_TIMEOUT", "60"))
    HEALTH_CHECK_INTERVAL: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))

    # UI constants
    APP_TITLE: str = "RAG Teaching Assistant"
    APP_ICON: str = "🎓"
    MAX_DISPLAY_SOURCES: int = 10
    DEFAULT_SOURCE_TYPES: list = ["pdf", "ppt", "video"]


config = UIConfig()
