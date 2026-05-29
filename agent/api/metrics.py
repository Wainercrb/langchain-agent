"""Simple in-memory request/error/latency counters.

LangSmith handles the heavy observability; this is a lightweight health-check
friendly endpoint with basic counters. Restart loss is acceptable per spec —
these are for operational awareness, not billing or auditing.
"""

import threading


class SimpleMetrics:
    """Thread-safe in-memory request/error/latency counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_count = 0
        self._error_count = 0
        self._total_latency_ms = 0.0

    def record_request(self, latency_ms: float) -> None:
        """Record a completed request with its latency."""
        with self._lock:
            self._request_count += 1
            self._total_latency_ms += latency_ms

    def record_error(self) -> None:
        """Record an error occurrence."""
        with self._lock:
            self._error_count += 1

    def snapshot(self) -> dict:
        """Return a point-in-time snapshot of all counters."""
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


# Singleton instance used by the /metrics endpoint
_metrics = SimpleMetrics()


def get_metrics() -> SimpleMetrics:
    """Return the global SimpleMetrics singleton."""
    return _metrics
