"""HTTP request counters — request_count, error_count, average latency.

Lightweight in-memory counters for operational awareness. LangSmith handles
the heavy observability; this is health-check friendly. Restart loss is
acceptable per spec — these are counters, not auditable records.
"""

import threading


class RequestMetrics:
    """Thread-safe in-memory request/error/latency counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_count = 0
        self._error_count = 0
        self._total_latency_ms = 0.0

    def record_request(self, latency_ms: float) -> None:
        """Record a completed request with its latency.

        Args:
            latency_ms: Total request latency in milliseconds.
        """
        with self._lock:
            self._request_count += 1
            self._total_latency_ms += latency_ms

    def record_error(self) -> None:
        """Record an error occurrence."""
        with self._lock:
            self._error_count += 1

    def snapshot(self) -> dict:
        """Return a point-in-time snapshot of the counters.

        Returns:
            Dictionary with `request_count`, `error_count`, and `avg_latency_ms`.
        """
        with self._lock:
            avg_latency = (
                round(self._total_latency_ms / self._request_count, 2)
                if self._request_count > 0
                else 0.0
            )
            return {
                "request_count": self._request_count,
                "error_count": self._error_count,
                "avg_latency_ms": avg_latency,
            }


_request_metrics = RequestMetrics()


def get_request_metrics() -> RequestMetrics:
    """Return the global RequestMetrics singleton."""
    return _request_metrics
