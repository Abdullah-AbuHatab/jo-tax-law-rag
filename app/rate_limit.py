"""A small in-process sliding-window rate limiter.

Limitation, on purpose: the counters live in this process's memory. With
one instance (Render free plan) that is exactly right; with several
instances each one counts separately, and a shared store (Redis) would be
needed instead.
"""

from __future__ import annotations

import hashlib
import ipaddress
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass


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


@dataclass(frozen=True)
class ClientIpSettings:
    """Where the caller's IP comes from.

    ``header``: a header set by the edge in front of the app and overwritten
    for every request (Cloudflare's ``cf-connecting-ip``); empty = not used.
    ``trusted_proxy_hops``: how many proxies append to X-Forwarded-For.
    """

    header: str = ""
    trusted_proxy_hops: int = 0


def _valid_ip(value: str | None) -> str | None:
    """The canonical form of ``value`` if it is one bare IP address, else None."""
    if not value:
        return None
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        return None


def client_ip(peer_host: str | None, headers: Mapping[str, str], settings: ClientIpSettings) -> str:
    """The caller's IP, in this order of precedence:

    1. ``settings.header``, when configured and it holds one valid IP. The
       edge (Cloudflare) overwrites it, so the client cannot forge it, and a
       forged X-Forwarded-For is then never looked at.
    2. X-Forwarded-For behind ``trusted_proxy_hops`` proxies that each append
       to it: the client is the N-th entry from the right; entries further
       left are client-controlled. With 0 hops the header is ignored.
    3. The socket peer.

    Header values that are not a single valid IP fall through to the next
    step, so junk in a header can never become a bucket key.
    """
    if settings.header:
        from_header = _valid_ip(headers.get(settings.header))
        if from_header:
            return from_header
    forwarded_for = headers.get("x-forwarded-for")
    if settings.trusted_proxy_hops > 0 and forwarded_for:
        parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
        if len(parts) >= settings.trusted_proxy_hops:
            from_forwarded = _valid_ip(parts[-settings.trusted_proxy_hops])
            if from_forwarded:
                return from_forwarded
    return peer_host or "unknown"


def client_network(ip: str) -> str:
    """What one client is, for limits: an IPv6 caller is its /64 network (any host
    controls at least a /64, so per-address keys would be unlimited), an
    IPv4-mapped IPv6 address is that IPv4 address, IPv4 is unchanged. Values that
    are not an IP (e.g. "unknown") pass through."""
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return ip
    if address.version == 4:
        return str(address)
    if address.ipv4_mapped is not None:
        return str(address.ipv4_mapped)
    return str(ipaddress.ip_network(f"{address}/64", strict=False))


def client_key(ip: str, salt: str) -> str:
    """A short pseudonymous id for ``ip``: logged instead of the raw address.

    The salt is random per process, so keys compare clients within one
    deploy only, and cannot be reversed with a table of all IPv4 addresses.
    """
    return hashlib.sha256((salt + ip).encode("utf-8")).hexdigest()[:12]
