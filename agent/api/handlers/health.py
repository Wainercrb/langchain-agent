"""Health endpoint — check service and database connectivity status."""

from datetime import datetime, timezone

from fastapi import APIRouter

from api.health_check import check_health
from infrastructure.logging import logger
from models import HealthResponse

router = APIRouter(prefix="/v1", tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=200,
    responses={
        500: {"model": "ErrorResponse", "description": "Health check failed"},
    },
)
async def health() -> HealthResponse:
    """
    Check service and database connectivity status.

    This endpoint performs a lightweight health check including:
    - Service availability
    - Database connectivity (Supabase/pgvector)
    - LangSmith tracing API connectivity (lightweight, no LLM invocation)
    - Embedding service connectivity

    Returns:
        HealthResponse containing:
            - status: "ok" if all checks pass, "error" if degraded
            - timestamp: Check timestamp
            - version: API version
            - db_connected: Database connectivity status
            - langsmith_connected: LangSmith tracing API availability
            - embedding_connected: Embedding service availability

    Raises:
        Returns degraded status rather than raising, to allow clients to assess
        partial system health (e.g., API available but database down).
    """
    logger.debug("Health check requested")
    health_info = await check_health()

    return HealthResponse(
        status=health_info["status"],
        timestamp=datetime.now(timezone.utc),
        db_connected=health_info.get("db_connected", False),
        llm_connected=health_info.get("llm_connected", False),
        langsmith_connected=health_info.get("langsmith_connected", False),
        embedding_connected=health_info.get("embedding_connected", False),
    )
