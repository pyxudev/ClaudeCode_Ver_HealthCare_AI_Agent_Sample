"""Redis: rate limiting and reservation idempotency.

Fail-open policy. Redis being down must not stop a patient from reaching a
doctor, so every helper degrades to "allow" and logs. The DB-level unique
constraints - not Redis - are what actually prevent a double booking; Redis
only saves the round trip.
"""

from __future__ import annotations

import logging
from typing import Optional

import redis.asyncio as redis

log = logging.getLogger("hdai.cache")


class Cache:
    def __init__(self, url: str) -> None:
        self._url = url
        self._client: Optional[redis.Redis] = None
        self.degraded = False

    async def connect(self) -> None:
        try:
            self._client = redis.from_url(
                self._url, encoding="utf-8", decode_responses=True,
                socket_connect_timeout=2.0, socket_timeout=2.0,
                health_check_interval=30,
            )
            await self._client.ping()
            self.degraded = False
            log.info("redis connected")
        except Exception as exc:  # noqa: BLE001
            self._client = None
            self.degraded = True
            log.warning("redis unavailable, continuing without it", extra={"error": str(exc)})

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    async def ping(self) -> bool:
        if self._client is None:
            return False
        try:
            await self._client.ping()
            self.degraded = False
            return True
        except Exception:  # noqa: BLE001
            self.degraded = True
            return False

    async def allow(self, key: str, limit: int, window_s: int = 60) -> bool:
        """Fixed-window rate limit. Returns True (allow) if Redis is down."""
        if self._client is None or limit <= 0:
            return True
        try:
            redis_key = f"rl:{key}"
            count = await self._client.incr(redis_key)
            if count == 1:
                await self._client.expire(redis_key, window_s)
            return count <= limit
        except Exception as exc:  # noqa: BLE001
            self.degraded = True
            log.warning("rate limit check failed, allowing", extra={"error": str(exc)})
            return True

    async def get(self, key: str) -> str | None:
        if self._client is None:
            return None
        try:
            return await self._client.get(key)
        except Exception:  # noqa: BLE001
            return None

    async def setex(self, key: str, value: str, ttl_s: int) -> None:
        if self._client is None:
            return
        try:
            await self._client.setex(key, ttl_s, value)
        except Exception:  # noqa: BLE001
            pass
