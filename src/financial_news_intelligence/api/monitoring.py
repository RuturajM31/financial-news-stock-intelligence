"""Dependency-free Prometheus metrics and operation instrumentation.

Purpose
-------
Collect bounded application metrics without importing native model libraries or
adding a third-party runtime dependency. The registry is process-local because
the current FastAPI runtime intentionally uses one worker. Docker and Kubernetes
stages can scrape each replica independently and aggregate in Prometheus.

Privacy and cardinality
-----------------------
Only fixed route templates, HTTP methods, status codes, and named operations are
used as labels. Request IDs, client addresses, article text, tickers, filenames,
and uploaded content are never metric labels.
"""

from __future__ import annotations

import functools
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, ParamSpec, TypeVar

from .logging_config import get_logger


P = ParamSpec("P")
R = TypeVar("R")
HISTOGRAM_BUCKETS = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
)


def _escape_label(value: str) -> str:
    """Escape one bounded Prometheus label value."""

    return value[:120].replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(values: Mapping[str, str]) -> str:
    """Render deterministic Prometheus labels."""

    if not values:
        return ""
    rendered = ",".join(
        f'{name}="{_escape_label(value)}"' for name, value in sorted(values.items())
    )
    return "{" + rendered + "}"


@dataclass
class HistogramState:
    """Store cumulative source observations for one label set."""

    observations: list[float] = field(default_factory=list)
    total: float = 0.0

    def observe(self, value: float) -> None:
        """Record one non-negative duration."""

        safe_value = max(0.0, float(value))
        self.observations.append(safe_value)
        self.total += safe_value


class MetricsRegistry:
    """Thread-safe process-local metrics registry with fixed metric families."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._started_at = time.monotonic()
        self._service = "financial-news-stock-intelligence-api"
        self._version = "unknown"
        self._ready = 0
        self._inflight = 0
        self._http_total: defaultdict[tuple[str, str, str], int] = defaultdict(int)
        self._http_duration: dict[tuple[str, str], HistogramState] = {}
        self._operation_total: defaultdict[tuple[str, str], int] = defaultdict(int)
        self._operation_duration: dict[str, HistogramState] = {}

    def configure(self, *, service: str, version: str) -> None:
        """Record immutable build labels used by downstream dashboards."""

        with self._lock:
            self._service = service[:120]
            self._version = version[:120]

    def set_ready(self, ready: bool) -> None:
        """Set the application lifecycle readiness gauge."""

        with self._lock:
            self._ready = 1 if ready else 0

    def request_started(self) -> None:
        """Increment the in-flight request gauge."""

        with self._lock:
            self._inflight += 1

    def request_finished(self) -> None:
        """Decrement the in-flight request gauge without going negative."""

        with self._lock:
            self._inflight = max(0, self._inflight - 1)

    def observe_http(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        """Record one completed HTTP request using bounded route labels."""

        safe_method = method.upper()[:16]
        safe_path = path[:120]
        safe_status = str(int(status_code))
        key = (safe_method, safe_path, safe_status)
        duration_key = (safe_method, safe_path)
        with self._lock:
            self._http_total[key] += 1
            histogram = self._http_duration.setdefault(
                duration_key,
                HistogramState(),
            )
            histogram.observe(duration_seconds)

    def observe_operation(
        self,
        *,
        operation: str,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        """Record one named service operation and duration."""

        safe_operation = operation[:80]
        safe_outcome = outcome[:20]
        with self._lock:
            self._operation_total[(safe_operation, safe_outcome)] += 1
            histogram = self._operation_duration.setdefault(
                safe_operation,
                HistogramState(),
            )
            histogram.observe(duration_seconds)

    @staticmethod
    def _render_histogram(
        name: str,
        help_text: str,
        states: Mapping[tuple[str, ...], HistogramState],
        label_names: tuple[str, ...],
    ) -> list[str]:
        """Render one histogram family from raw observations."""

        lines = [f"# HELP {name} {help_text}", f"# TYPE {name} histogram"]
        for key in sorted(states):
            state = states[key]
            base_labels = dict(zip(label_names, key))
            for bucket in HISTOGRAM_BUCKETS:
                count = sum(1 for value in state.observations if value <= bucket)
                labels = dict(base_labels)
                labels["le"] = f"{bucket:g}"
                lines.append(f"{name}_bucket{_labels(labels)} {count}")
            infinite_labels = dict(base_labels)
            infinite_labels["le"] = "+Inf"
            lines.append(
                f"{name}_bucket{_labels(infinite_labels)} {len(state.observations)}"
            )
            lines.append(f"{name}_sum{_labels(base_labels)} {state.total:.9f}")
            lines.append(f"{name}_count{_labels(base_labels)} {len(state.observations)}")
        return lines

    def render(self) -> str:
        """Return a stable Prometheus text exposition document."""

        with self._lock:
            http_total = dict(self._http_total)
            http_duration = {
                key: HistogramState(list(value.observations), value.total)
                for key, value in self._http_duration.items()
            }
            operation_total = dict(self._operation_total)
            operation_duration = {
                (key,): HistogramState(list(value.observations), value.total)
                for key, value in self._operation_duration.items()
            }
            service = self._service
            version = self._version
            ready = self._ready
            inflight = self._inflight
            uptime = max(0.0, time.monotonic() - self._started_at)

        lines = [
            "# HELP fni_build_info Static application build information.",
            "# TYPE fni_build_info gauge",
            f"fni_build_info{_labels({'service': service, 'version': version})} 1",
            "# HELP fni_process_uptime_seconds Process uptime in seconds.",
            "# TYPE fni_process_uptime_seconds gauge",
            f"fni_process_uptime_seconds {uptime:.3f}",
            "# HELP fni_application_ready Application lifecycle readiness.",
            "# TYPE fni_application_ready gauge",
            f"fni_application_ready {ready}",
            "# HELP fni_http_requests_in_flight Current in-flight HTTP requests.",
            "# TYPE fni_http_requests_in_flight gauge",
            f"fni_http_requests_in_flight {inflight}",
            "# HELP fni_http_requests_total Completed HTTP requests.",
            "# TYPE fni_http_requests_total counter",
        ]
        for method, path, status in sorted(http_total):
            lines.append(
                "fni_http_requests_total"
                + _labels({"method": method, "path": path, "status": status})
                + f" {http_total[(method, path, status)]}"
            )
        lines.extend(
            self._render_histogram(
                "fni_http_request_duration_seconds",
                "HTTP request duration in seconds.",
                http_duration,
                ("method", "path"),
            )
        )
        lines.extend(
            [
                "# HELP fni_service_operations_total Completed service operations.",
                "# TYPE fni_service_operations_total counter",
            ]
        )
        for operation, outcome in sorted(operation_total):
            lines.append(
                "fni_service_operations_total"
                + _labels({"operation": operation, "outcome": outcome})
                + f" {operation_total[(operation, outcome)]}"
            )
        lines.extend(
            self._render_histogram(
                "fni_service_operation_duration_seconds",
                "Service operation duration in seconds.",
                operation_duration,
                ("operation",),
            )
        )
        return "\n".join(lines) + "\n"


_REGISTRY = MetricsRegistry()


def get_metrics_registry() -> MetricsRegistry:
    """Return the process-global registry used by the single API worker."""

    return _REGISTRY


def reset_metrics_registry() -> MetricsRegistry:
    """Replace the global registry for deterministic tests only."""

    global _REGISTRY
    _REGISTRY = MetricsRegistry()
    return _REGISTRY


def route_template(scope: Mapping[str, Any]) -> str:
    """Return a fixed FastAPI route template or one low-cardinality fallback."""

    route = scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and path.startswith("/"):
        return path[:120]
    return "unmatched"


def observe_operation(operation: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate one synchronous service boundary with metrics and JSON logs."""

    if not operation or len(operation) > 80:
        raise ValueError("operation must contain between 1 and 80 characters")

    def decorator(function: Callable[P, R]) -> Callable[P, R]:
        logger = get_logger(function.__module__)

        @functools.wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            started = time.perf_counter()
            logger.info(
                "Service operation started.",
                extra={"marker": "STARTED", "operation": operation},
            )
            try:
                result = function(*args, **kwargs)
            except Exception:
                duration = max(0.0, time.perf_counter() - started)
                get_metrics_registry().observe_operation(
                    operation=operation,
                    outcome="failed",
                    duration_seconds=duration,
                )
                logger.exception(
                    "Service operation failed.",
                    extra={
                        "marker": "FAILED",
                        "operation": operation,
                        "outcome": "failed",
                        "duration_ms": round(duration * 1000, 3),
                    },
                )
                raise
            duration = max(0.0, time.perf_counter() - started)
            get_metrics_registry().observe_operation(
                operation=operation,
                outcome="passed",
                duration_seconds=duration,
            )
            logger.info(
                "Service operation completed.",
                extra={
                    "marker": "PASSED",
                    "operation": operation,
                    "outcome": "passed",
                    "duration_ms": round(duration * 1000, 3),
                },
            )
            return result

        return wrapped

    return decorator
