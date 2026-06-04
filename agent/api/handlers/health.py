"""Health endpoint — verify service readiness using the single source of truth.

Delegates all checks to functions from ``observability.health.checks``
(the same functions used by the background monitoring scheduler) so the
endpoint and monitoring always share identical logic.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

from container import embeddings, observability, vector_store
from config import settings
from models import HealthResponse
from observability.health.checks import check_database, check_embeddings_service

router = APIRouter(prefix="/v1", tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=200,
)
async def health() -> HealthResponse:
    """Check service readiness.

    Runs real connectivity checks for DB, observability backend, and embeddings.
    LLM check is configuration-only (no token costs).
    """
    db_result = await check_database(vector_store)
    observability_result = await observability.health_check()
    embedding_result = await check_embeddings_service(embeddings)

    all_ok = db_result.ok and observability_result.ok and embedding_result.ok

    return HealthResponse(
        status="ok" if all_ok else "degraded",
        timestamp=datetime.now(timezone.utc),
        db_connected=db_result.ok,
        llm_connected=bool(
            settings.google_api_key
            or settings.openrouter_api_key
            or settings.openai_api_key
        ),
        observability_connected=observability_result.ok,
        embedding_connected=embedding_result.ok,
    )
