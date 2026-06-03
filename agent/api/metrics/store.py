"""Unified metrics store — consolidates request and LLM-usage counters.

Single thread-safe store for all operational metrics. Replaces the separate
RequestMetrics and LLMUsageMetrics singletons.
"""

import threading
from typing import Any, Optional


class MetricsStore:
    """Thread-safe in-memory store for HTTP request and LLM token counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_count = 0
        self._error_count = 0
        self._total_latency_ms = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._llm_record_count = 0

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

    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage for a completed LLM call.

        Negative values are ignored — they indicate a provider metadata bug
        and must not corrupt the counters.

        Args:
            input_tokens: Prompt/input tokens consumed.
            output_tokens: Completion/output tokens generated.
        """
        if input_tokens < 0 or output_tokens < 0:
            return
        with self._lock:
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._llm_record_count += 1

    def snapshot(self, decision_tracker: Optional[Any] = None) -> dict:
        """Return a point-in-time snapshot of all counters.

        Args:
            decision_tracker: Optional DecisionTracker instance to include AI
                decision aggregates (total_decisions, decisions_evicted, store_size)
                under the `ai_decisions` key.

        Returns:
            Dictionary with request counters, token counters, and optionally
            decision aggregates.
        """
        with self._lock:
            avg_latency = (
                round(self._total_latency_ms / self._request_count, 2)
                if self._request_count > 0
                else 0.0
            )
            total_tokens = self._total_input_tokens + self._total_output_tokens
            avg_tokens = (
                round(total_tokens / self._llm_record_count, 2)
                if self._llm_record_count > 0
                else 0.0
            )
            snapshot = {
                "request_count": self._request_count,
                "error_count": self._error_count,
                "avg_latency_ms": avg_latency,
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "avg_tokens_per_request": avg_tokens,
            }

        if decision_tracker is not None:
            snapshot["ai_decisions"] = {
                "total_decisions": decision_tracker.size,
                "decisions_evicted": decision_tracker.eviction_count,
                "store_size": decision_tracker.size,
            }

        return snapshot


_metrics_store = MetricsStore()


def get_metrics_store() -> MetricsStore:
    """Return the global MetricsStore singleton."""
    return _metrics_store
