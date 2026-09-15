"""Rate limiting and response caching for the synthesis endpoints."""

from __future__ import annotations

import time
from collections import OrderedDict

from fastapi import HTTPException


class RateLimiter:
    """Token bucket, shared across all endpoints.

    Synthesis is expensive and unbounded concurrency on a single GPU just makes
    every request slow, so excess load is rejected rather than queued.
    """

    def __init__(self, rate: float = 10.0):
        self._rate = rate
        self._tokens = rate
        self._last = time.monotonic()

    def __call__(self) -> None:
        now = time.monotonic()
        self._tokens = min(self._rate, self._tokens + self._rate * (now - self._last))
        self._last = now
        if self._tokens < 1.0:
            raise HTTPException(
                status_code=429, detail="Rate limit exceeded. Try again shortly.",
            )
        self._tokens -= 1.0


class LRUCache:
    """Fixed-size LRU cache of rendered audio, keyed by query string.

    Slider UIs re-request the same settings constantly (A/B-ing two positions,
    re-triggering the same sound), and synthesis is deterministic, so the hit
    rate is high. Cleared whenever the loaded model changes.
    """

    def __init__(self, max_size: int = 100):
        self._max_size = max_size
        self._store: OrderedDict[str, bytes] = OrderedDict()

    def get(self, key: str) -> bytes | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, value: bytes) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        elif len(self._store) >= self._max_size:
            self._store.popitem(last=False)
        self._store[key] = value

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
