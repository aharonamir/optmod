"""
Short-lived "this model just failed" memory.

A provider model that fails (404 slug gone, 5xx, timeout, 429) is skipped for a
TTL that depends on the error type, so the next requests do not walk into the
same failure. In-memory only; wiped on restart. Never raises.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

DEFAULTS: dict = {
    "enabled": True,
    "max_ttl_s": 86400.0,       # cap on a provider-advertised reset (Retry-After / X-RateLimit-Reset)
    "ttl_s": {
        "not_found":  3600.0,   # model slug no longer exists — will not recover soon
        "server":      120.0,
        "timeout":     120.0,
        "rate_limit":   30.0,
        "auth":        600.0,   # one model's key/access is rejected; others may be fine
    },
}
# bad_request is never marked: it points at the request, not at the model.


class ModelHealth:
    def __init__(self, cfg: dict | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", DEFAULTS["enabled"]))
        self._ttl = {**DEFAULTS["ttl_s"], **(cfg.get("ttl_s") or {})}
        self._max_ttl = float(cfg.get("max_ttl_s", DEFAULTS["max_ttl_s"]))
        self._clock = clock
        self._lock = threading.Lock()
        self._bad: dict[str, tuple[float, str]] = {}   # model -> (until, error_type)

    def mark_failure(self, model: str, error_type: str | None,
                     retry_after_s: float | None = None) -> None:
        """retry_after_s: provider-advertised wait; replaces the configured TTL (capped by max_ttl_s)."""
        if not self.enabled or not error_type:
            return
        ttl = float(self._ttl.get(error_type, 0) or 0)
        if ttl <= 0:
            return
        if retry_after_s is not None and retry_after_s > 0:
            ttl = min(float(retry_after_s), self._max_ttl)
        with self._lock:
            self._bad[model] = (self._clock() + ttl, error_type)

    def mark_success(self, model: str) -> None:
        with self._lock:
            self._bad.pop(model, None)

    def is_bad(self, model: str) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            entry = self._bad.get(model)
            if entry is None:
                return False
            if self._clock() >= entry[0]:
                del self._bad[model]
                return False
            return True

    def clear(self) -> None:
        with self._lock:
            self._bad.clear()

    def snapshot(self) -> dict:
        now = self._clock()
        with self._lock:
            return {
                m: {"error_type": et, "remaining_s": round(until - now, 1)}
                for m, (until, et) in self._bad.items() if until > now
            }
