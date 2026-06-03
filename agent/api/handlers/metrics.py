"""Metrics endpoint — return operational counters since process startup."""

from fastapi import APIRouter, Depends

from api.api_responses import build_metrics_response
from config import get_langsmith_dashboard_url
from container import decision_tracker
from api.metrics_store import build_metrics_snapshot
from models import MetricsResponse

router = APIRouter(prefix="/v1", tags=["metrics"])


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    status_code=200,
)
async def metrics_endpoint(
    tracker=Depends(lambda: decision_tracker),
) -> MetricsResponse:
    """
    Return lightweight operational counters since process startup.

    LangSmith handles the comprehensive observability dashboards;
    this endpoint provides simple request/error/latency counters
    that are health-check friendly.

    Args:
        tracker: Injected DecisionTracker — used to attach AI decision
            aggregates (total_decisions, decisions_evicted, store_size)
            to the snapshot.

    Returns:
        MetricsResponse with request_count, error_count, avg_latency_ms,
        and langsmith_dashboard_url (if tracing is configured).
    """
    data = build_metrics_snapshot()
    data["ai_decisions"] = {
        "total_decisions": tracker.size,
        "decisions_evicted": tracker.eviction_count,
    }

    langsmith_url = get_langsmith_dashboard_url()

    return build_metrics_response(data=data, langsmith_dashboard_url=langsmith_url)
