from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """In-process sliding-window rate limiter, keyed by API key.

    Good enough for a single-instance deployment (matches the shipped
    docker-compose, one container). Scaling to multiple replicas needs a
    shared store (e.g. Redis) instead of this in-memory dict, since each
    replica would otherwise track its own independent window.
    """

    def __init__(self):
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit_per_minute: int) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds). Records the hit iff allowed."""
        now = time.monotonic()
        window_start = now - 60.0
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] < window_start:
                hits.popleft()
            if len(hits) >= limit_per_minute:
                retry_after = 60.0 - (now - hits[0])
                return False, max(retry_after, 0.0)
            hits.append(now)
            return True, 0.0


rate_limiter = RateLimiter()
