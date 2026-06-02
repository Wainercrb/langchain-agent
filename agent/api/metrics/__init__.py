"""Metrics aggregator — combines request and LLM-usage snapshots.

This package replaces the legacy monolithic `api.metrics` module. Each metric
family lives in its own submodule with a single responsibility:

- `request`: HTTP request counters (request_count, error_count, avg_latency_ms)
- `llm_usage`: LLM token counters (input/output tokens, avg per request)

The optional `decision_tracker` parameter on `build_metrics_snapshot` lets the
route handler attach AI decision aggregates without coupling the metrics
classes to the DecisionTracker.
"""

from typing import Any, Optional

from .llm_usage import LLMUsageMetrics, get_llm_usage_metrics
from .request import RequestMetrics, get_request_metrics

__all__ = [
    "LLMUsageMetrics",
    "RequestMetrics",
    "build_metrics_snapshot",
    "get_llm_usage_metrics",
    "get_request_metrics",
]


def build_metrics_snapshot(decision_tracker: Optional[Any] = None) -> dict:
    """Combine request and LLM-usage snapshots, optionally with decision aggregates.

    Args:
        decision_tracker: Optional DecisionTracker instance to include AI
            decision aggregates (total_decisions, decisions_evicted, store_size)
            under the `ai_decisions` key.

    Returns:
        Dictionary merging request counters, token counters, and (optionally)
        decision aggregates. Backward-compatible with the legacy
        `SimpleMetrics.snapshot()` output shape.
    """
    snapshot = {
        **get_request_metrics().snapshot(),
        **get_llm_usage_metrics().snapshot(),
    }

    if decision_tracker is not None:
        snapshot["ai_decisions"] = {
            "total_decisions": decision_tracker.size,
            "decisions_evicted": decision_tracker.eviction_count,
            "store_size": decision_tracker.size,
        }

    return snapshot
