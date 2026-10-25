"""In-process abuse and cost guards that sit in front of AnswerService.

* ``DailyBudget``: a global allowance per UTC day (embedding calls, LLM calls),
  so rotating IPs cannot drain the owner's free-tier quotas.
* ``PerKeyDailyLimit``: a per-client allowance per UTC day, bounded in memory.
* ``ConcurrencyGate``: caps the /ask requests in flight, so slow LLM calls
  cannot hold every worker thread.
* ``AnswerCache``: repeated questions are answered without spending quota.

Same limitation as the rate limiter, on purpose: the state lives in this
process's memory. With one instance (Render free plan) that is exactly
right; a restart resets the counters, which errs on the side of serving.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from datetime import time as dt_time
from types import TracebackType

from app.arabic import normalize_arabic
from app.schemas import AskResponse


def utc_now() -> datetime:
    return datetime.now(UTC)


def _utc(now: datetime) -> datetime:
    # A naive datetime is taken as UTC, not as the server's local time.
    return now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)


def seconds_until_utc_midnight(now: datetime) -> int:
    """Whole seconds until the next 00:00 UTC (at least 1), for Retry-After."""
    current = _utc(now)
    midnight = datetime.combine(current.date() + timedelta(days=1), dt_time.min, tzinfo=UTC)
    return max(1, math.ceil((midnight - current).total_seconds()))


class DailyBudget:
    """At most ``limit`` units per UTC day, shared by every client. Thread-safe."""

    def __init__(self, limit: int, clock: Callable[[], datetime] = utc_now) -> None:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        self._limit = limit
        self._clock = clock
        self._lock = threading.Lock()
        self._day: date = _utc(clock()).date()
        self._used = 0

    def _roll(self) -> datetime:
        now = self._clock()
        today = _utc(now).date()
        if today != self._day:
            self._day, self._used = today, 0
        return now

    def try_consume(self, n: int = 1) -> bool:
        """Spend ``n`` units if all of them are left today; otherwise spend nothing."""
        if n < 1:
            raise ValueError("n must be >= 1")
        with self._lock:
            self._roll()
            if self._used + n > self._limit:
                return False
            self._used += n
            return True

    def remaining(self) -> int:
        with self._lock:
            self._roll()
            return self._limit - self._used

    def seconds_until_reset(self) -> int:
        return seconds_until_utc_midnight(self._clock())


class PerKeyDailyLimit:
    """At most ``limit`` hits per key per UTC day. Thread-safe.

    Memory is bounded: at most ``max_keys`` keys are tracked, and the least
    recently seen key is forgotten first (it then starts again from zero).
    """

    def __init__(self, limit: int, clock: Callable[[], datetime] = utc_now, max_keys: int = 10_000) -> None:
        if limit < 1 or max_keys < 1:
            raise ValueError("limit and max_keys must be >= 1")
        self._limit = limit
        self._clock = clock
        self._max_keys = max_keys
        self._lock = threading.Lock()
        self._day: date = _utc(clock()).date()
        self._counts: OrderedDict[str, int] = OrderedDict()

    def allow(self, key: str) -> bool:
        """Record a hit for ``key`` if it is under today's limit."""
        with self._lock:
            today = _utc(self._clock()).date()
            if today != self._day:
                self._day = today
                self._counts.clear()
            used = self._counts.get(key, 0)
            if used >= self._limit:
                self._counts.move_to_end(key)
                return False
            self._counts[key] = used + 1
            self._counts.move_to_end(key)
            while len(self._counts) > self._max_keys:
                self._counts.popitem(last=False)
            return True

    def refund(self, key: str) -> None:
        """Give back one hit, for a request that was counted but never served
        (BUSY, budget exhausted). Never goes below zero."""
        with self._lock:
            today = _utc(self._clock()).date()
            if today != self._day:
                return  # counted on a previous day: that count is already gone
            used = self._counts.get(key, 0)
            if used > 0:
                self._counts[key] = used - 1

    def seconds_until_reset(self) -> int:
        return seconds_until_utc_midnight(self._clock())

    def __len__(self) -> int:
        return len(self._counts)


class SingleFlight:
    """At most one computation per key at a time; identical requests wait for it.

    ``lead(key)`` is True for the caller that must compute (and then call
    ``finish(key)``, after storing its result). Others call ``wait(key)``: it
    blocks until the leader finishes, but at most ``max_wait_seconds``, and
    only ``max_waiters`` callers in total may wait (each one holds a worker
    thread), so a flood of identical questions cannot exhaust the pool.
    """

    def __init__(self, max_wait_seconds: float = 30.0, max_waiters: int = 8) -> None:
        self._max_wait = max_wait_seconds
        self._max_waiters = max_waiters
        self._lock = threading.Lock()
        self._in_flight: dict[str, threading.Event] = {}
        self._waiters = 0

    def lead(self, key: str) -> bool:
        with self._lock:
            if key in self._in_flight:
                return False
            self._in_flight[key] = threading.Event()
            return True

    def wait(self, key: str) -> bool:
        """True when nothing is (any longer) in flight for ``key``."""
        with self._lock:
            event = self._in_flight.get(key)
            if event is None:
                return True
            if self._waiters >= self._max_waiters:
                return False
            self._waiters += 1
        try:
            return event.wait(self._max_wait)
        finally:
            with self._lock:
                self._waiters -= 1

    def finish(self, key: str) -> None:
        with self._lock:
            event = self._in_flight.pop(key, None)
        if event is not None:
            event.set()

    def waiting(self) -> int:
        with self._lock:
            return self._waiters


class GateFull(Exception):
    """Raised by ``with gate:`` when every slot is taken."""


class ConcurrencyGate:
    """At most ``max_in_flight`` holders at once; never waits for a slot.

    ``with gate:`` takes a slot or raises ``GateFull`` and always gives the
    slot back on exit. ``acquire_nowait``/``release`` are the same, by hand.
    """

    def __init__(self, max_in_flight: int) -> None:
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be >= 1")
        self._slots = threading.BoundedSemaphore(max_in_flight)

    def acquire_nowait(self) -> bool:
        return self._slots.acquire(blocking=False)

    def release(self) -> None:
        self._slots.release()

    def __enter__(self) -> ConcurrencyGate:
        if not self.acquire_nowait():
            raise GateFull
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.release()


class AnswerCache:
    """LRU cache of successful /ask responses with a TTL. Thread-safe.

    Responses that carry an ``error_code`` are never stored. ``max_entries=0``
    disables the cache.
    """

    def __init__(
        self, max_entries: int = 500, ttl_seconds: int = 86400, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if max_entries < 0 or ttl_seconds < 1:
            raise ValueError("max_entries must be >= 0 and ttl_seconds >= 1")
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[float, AskResponse]] = OrderedDict()

    @staticmethod
    def key_for(question: str, law_id: str | None, top_k: int) -> str:
        """Spelling variants of one question (diacritics, hamza forms, spaces) share a key."""
        material = json.dumps([normalize_arabic(question), law_id, top_k], ensure_ascii=False)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def get(self, key: str) -> AskResponse | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, value = entry
            if self._clock() - stored_at >= self._ttl:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return value

    def put(self, key: str, value: AskResponse) -> None:
        if self._max_entries == 0 or value.error_code is not None:
            return
        with self._lock:
            self._entries[key] = (self._clock(), value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
