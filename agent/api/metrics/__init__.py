"""Metrics aggregator — unified MetricsStore for all operational counters.

This package provides a single MetricsStore that consolidates:
- HTTP request counters (request_count, error_count, avg_latency_ms)
- LLM token counters (input/output tokens, avg per request)

Backward-compatible re-exports are provided for legacy code:
- `get_request_metrics()` returns the unified store (has record_request/record_error)
- `get_llm_usage_metrics()` returns the unified store (has record_tokens)
- `build_metrics_snapshot()` delegates to the unified store's snapshot()
"""

from typing import Any, Optional

from .store import MetricsStore, get_metrics_store

__all__ = [
    "MetricsStore",
    "build_metrics_snapshot",
    "get_metrics_store",
    "get_llm_usage_metrics",
    "get_request_metrics",
]


def get_request_metrics() -> MetricsStore:
    """Return the global MetricsStore singleton (backward-compatible)."""
    return get_metrics_store()


def get_llm_usage_metrics() -> MetricsStore:
    """Return the global MetricsStore singleton (backward-compatible)."""
    return get_metrics_store()


def build_metrics_snapshot(decision_tracker: Optional[Any] = None) -> dict:
    """Return a snapshot of all metrics, optionally with decision aggregates.

    Args:
        decision_tracker: Optional DecisionTracker instance to include AI
            decision aggregates under the `ai_decisions` key.

    Returns:
        Dictionary merging request counters, token counters, and (optionally)
        decision aggregates.
    """
    return get_metrics_store().snapshot(decision_tracker=decision_tracker)
