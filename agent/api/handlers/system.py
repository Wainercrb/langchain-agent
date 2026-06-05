"""System status endpoint — consolidated health + metrics in a single call.

Replaces the need for separate /v1/health and /v1/metrics calls from the dashboard.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from api.metrics_store import build_metrics_snapshot
from config import settings
from container import decision_tracker, embeddings, llm, observability, vector_store
from models import SystemStatusResponse
from observability.health.checks import check_database, check_embeddings_service

router = APIRouter(prefix="/v1", tags=["system"])


@router.get(
    "/system/status",
    response_model=SystemStatusResponse,
    status_code=200,
)
async def system_status(
    tracker=Depends(lambda: decision_tracker),
) -> SystemStatusResponse:
    """Return combined health checks and operational metrics.

    Runs real connectivity checks for DB, observability, and embeddings,
    then attaches request/error/latency/token counters.
    """
    db_result = await check_database(vector_store)
    observability_result = await observability.health_check()
    embedding_result = await check_embeddings_service(embeddings)
    metrics = build_metrics_snapshot()

    all_ok = db_result.ok and observability_result.ok and embedding_result.ok

    return SystemStatusResponse(
        status="ok" if all_ok else "degraded",
        timestamp=datetime.now(timezone.utc),
        db_connected=db_result.ok,
        llm_connected=bool(
            settings.google_api_key
            or settings.openrouter_api_key
            or settings.openai_api_key
        ),
        embedding_connected=embedding_result.ok,
        request_count=metrics["request_count"],
        error_count=metrics["error_count"],
        avg_latency_ms=metrics["avg_latency_ms"],
        total_input_tokens=metrics["total_input_tokens"],
        total_output_tokens=metrics["total_output_tokens"],
        circuits=llm.circuit_status() if hasattr(llm, "circuit_status") else {},
    )
