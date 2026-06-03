"""Health endpoint — verify service readiness without invoking any LLM."""

from datetime import datetime, timezone

from fastapi import APIRouter

from config import settings
from infrastructure.container import vector_store
from models import HealthResponse

router = APIRouter(prefix="/v1", tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=200,
)
async def health() -> HealthResponse:
    """Check service readiness.

    Only DB performs an actual connectivity ping. LLM, LangSmith, and
    embedding checks are configuration-only — no LLM invocations, no
    token costs, no external API calls.
    """
    db_connected = False
    try:
        db_connected = bool(await vector_store.health_check())
    except Exception:
        pass

    return HealthResponse(
        status="ok" if db_connected else "error",
        timestamp=datetime.now(timezone.utc),
        db_connected=db_connected,
        llm_connected=bool(
            settings.google_api_key
            or settings.openrouter_api_key
            or settings.openai_api_key
        ),
        langsmith_connected=bool(
            settings.enable_langsmith_tracing
            and settings.langsmith_api_key
        ),
        embedding_connected=bool(settings.google_api_key),
    )
