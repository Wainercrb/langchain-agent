"""Health endpoint — verify service readiness using the single source of truth.

Delegates all checks to HealthVerifier (the same instance used by the
background monitoring scheduler) so the endpoint and monitoring always
share identical logic.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

from container import _health_verifier
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

    Runs real connectivity checks for DB, LangSmith, and embeddings.
    LLM check is configuration-only (no token costs).
    """
    db_ok, _ = await _health_verifier.check_db()
    langsmith_ok, _ = await _health_verifier.check_langsmith()
    embedding_ok, _ = await _health_verifier.check_embeddings()

    all_ok = db_ok and langsmith_ok and embedding_ok

    return HealthResponse(
        status="ok" if all_ok else "degraded",
        timestamp=datetime.now(timezone.utc),
        db_connected=db_ok,
        llm_connected=bool(
            settings.google_api_key
            or settings.openrouter_api_key
            or settings.openai_api_key
        ),
        langsmith_connected=langsmith_ok,
        embedding_connected=embedding_ok,
    )
