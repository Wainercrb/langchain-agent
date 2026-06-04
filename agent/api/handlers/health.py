"""Health endpoint — verify service readiness using the single source of truth.

Delegates all checks to HealthVerifier (the same instance used by the
background monitoring scheduler) so the endpoint and monitoring always
share identical logic.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

from container import _health_verifier, observability
from config import settings
from models import HealthResponse

router = APIRouter(prefix="/v1", tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=200,
)
async def health() -> HealthResponse:
    """Check service readiness using HealthVerifier (single source of truth).

    Runs real connectivity checks for DB, observability backend, and embeddings.
    LLM check is configuration-only (no token costs).
    """
    db_result = await _health_verifier.check_db()
    observability_result = await observability.health_check()
    embedding_result = await _health_verifier.check_embeddings()

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
