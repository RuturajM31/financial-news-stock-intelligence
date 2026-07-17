"""Deterministic in-process fixed-window request limiting.

The limiter protects the local single-worker FastAPI phase. Docker and
Kubernetes phases must add a shared reverse-proxy or distributed limiter before
running multiple API replicas.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .errors import ApiProblem


@dataclass
class _Window:
    started_at: float
    request_count: int


class FixedWindowRateLimiter:
    """Limit requests per client-and-route key within a fixed time window."""

    def __init__(self, maximum_requests: int, window_seconds: int) -> None:
        if maximum_requests < 1 or window_seconds < 1:
            raise ValueError("Rate-limit values must be positive.")
        self.maximum_requests = maximum_requests
        self.window_seconds = window_seconds
        self._windows: dict[str, _Window] = {}
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> None:
        """Allow one request or raise a clear 429 failure."""

        current_time = time.monotonic() if now is None else now
        with self._lock:
            active_window = self._windows.get(key)
            if (
                active_window is None
                or current_time - active_window.started_at >= self.window_seconds
            ):
                self._windows[key] = _Window(current_time, 1)
                self._cleanup(current_time)
                return
            if active_window.request_count >= self.maximum_requests:
                raise ApiProblem(
                    status_code=429,
                    error_code="rate_limit_exceeded",
                    what_failed="The API request was rejected.",
                    where_failed="Request rate limiter",
                    why_failed=(
                        "The client exceeded the configured request count "
                        "inside the current time window."
                    ),
                    safe_next_step=(
                        f"Wait {self.window_seconds} seconds and retry once."
                    ),
                )
            active_window.request_count += 1

    def _cleanup(self, now: float) -> None:
        """Remove expired windows so client keys do not grow without bound."""

        expired_keys = [
            key
            for key, window in self._windows.items()
            if now - window.started_at >= self.window_seconds * 2
        ]
        for key in expired_keys:
            self._windows.pop(key, None)
