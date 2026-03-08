"""
Redis caching module.

Provides a ``RedisCache`` class for question→answer caching
with configurable TTL and graceful fallback when Redis is unavailable.
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional

import redis

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RedisCache:
    """
    Redis-backed cache for RAG responses.

    Caches serialized JSON keyed by a deterministic hash
    of the question string. Gracefully degrades to a no-op
    if the Redis server is unreachable.

    Attributes:
        client: A Redis client instance (or None on connection failure).
        ttl: Time-to-live for cache entries in seconds.
        prefix: Key prefix for namespacing.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl: Optional[int] = None,
        prefix: str = "rag:",
    ) -> None:
        settings = get_settings()
        self.ttl = ttl or settings.cache_ttl_seconds
        self.prefix = prefix
        self.client: Optional[redis.Redis] = None

        url = redis_url or settings.redis_url
        try:
            self.client = redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=3,
            )
            self.client.ping()
            logger.info("Redis connection established", extra={"url": url})
        except (redis.ConnectionError, redis.TimeoutError) as exc:
            logger.warning(
                "Redis unavailable — caching disabled",
                extra={"error": str(exc)},
            )
            self.client = None

    def _make_key(self, question: str) -> str:
        """
        Generate a deterministic cache key from a question.

        Args:
            question: The raw question string.

        Returns:
            A prefixed SHA-256 hash key.
        """
        digest = hashlib.sha256(question.strip().lower().encode()).hexdigest()
        return f"{self.prefix}{digest}"

    def get(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached response.

        Args:
            question: The question to look up.

        Returns:
            The cached response dict, or ``None`` on miss / error.
        """
        if self.client is None:
            return None

        try:
            key = self._make_key(question)
            data = self.client.get(key)
            if data:
                logger.info("Cache HIT", extra={"key": key})
                return json.loads(data)
            logger.debug("Cache MISS", extra={"key": key})
            return None
        except Exception as exc:
            logger.warning("Cache get error", extra={"error": str(exc)})
            return None

    def set(self, question: str, response: Dict[str, Any]) -> None:
        """
        Store a response in the cache.

        Args:
            question: The question string (key source).
            response: The response dict to cache.
        """
        if self.client is None:
            return

        try:
            key = self._make_key(question)
            self.client.setex(key, self.ttl, json.dumps(response))
            logger.info("Cache SET", extra={"key": key, "ttl": self.ttl})
        except Exception as exc:
            logger.warning("Cache set error", extra={"error": str(exc)})

    def invalidate(self, question: str) -> None:
        """
        Remove a specific entry from the cache.

        Args:
            question: The question whose cache entry to remove.
        """
        if self.client is None:
            return

        try:
            key = self._make_key(question)
            self.client.delete(key)
            logger.info("Cache INVALIDATED", extra={"key": key})
        except Exception as exc:
            logger.warning("Cache invalidate error", extra={"error": str(exc)})

    def flush(self) -> None:
        """Flush all keys with the configured prefix."""
        if self.client is None:
            return

        try:
            cursor = 0
            while True:
                cursor, keys = self.client.scan(
                    cursor=cursor, match=f"{self.prefix}*", count=100
                )
                if keys:
                    self.client.delete(*keys)
                if cursor == 0:
                    break
            logger.info("Cache flushed")
        except Exception as exc:
            logger.warning("Cache flush error", extra={"error": str(exc)})

    def is_available(self) -> bool:
        """Check if Redis is reachable."""
        if self.client is None:
            return False
        try:
            self.client.ping()
            return True
        except Exception:
            return False
