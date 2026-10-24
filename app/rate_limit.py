"""A small in-process sliding-window rate limiter.

Limitation, on purpose: the counters live in this process's memory. With
one instance (Render free plan) that is exactly right; with several
instances each one counts separately, and a shared store (Redis) would be
needed instead.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._last_sweep = clock()

    def allow(self, key: str) -> tuple[bool, int]:
        """Record a hit for ``key``. Returns (allowed, seconds until a slot frees up)."""
        now = self._clock()
        with self._lock:
            self._sweep(now)
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                retry_after = int(self._window - (now - hits[0])) + 1
                return False, retry_after
            hits.append(now)
            return True, 0

    def _sweep(self, now: float) -> None:
        # Forget idle clients once per window so memory stays bounded.
        if now - self._last_sweep < self._window:
            return
        self._last_sweep = now
        for key in [k for k, hits in self._hits.items() if not hits or now - hits[-1] >= self._window]:
            del self._hits[key]


def client_ip(peer_host: str | None, forwarded_for: str | None, trusted_proxy_hops: int) -> str:
    """The caller's IP.

    With ``trusted_proxy_hops = 0`` the socket peer is used and
    X-Forwarded-For is ignored (anyone can send that header). Behind N
    trusted proxies that each append to X-Forwarded-For, the client is the
    N-th entry from the right; entries further left are client-controlled.
    """
    if trusted_proxy_hops > 0 and forwarded_for:
        parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
        if len(parts) >= trusted_proxy_hops:
            return parts[-trusted_proxy_hops]
    return peer_host or "unknown"
