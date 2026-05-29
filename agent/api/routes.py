import threading
import time
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from rag.core.chain import RAGChain

from .dependencies import check_health, get_feedback_service, get_rag_chain
from config import settings
from models import ChatRequest, ChatResponse, ErrorResponse, FeedbackRequest, HealthResponse, MetricsResponse
from services.container import logger


# ── Simple In-Memory Metrics ─────────────────────────────────────────
# LangSmith handles the heavy observability; this is a lightweight health-
# check friendly endpoint with basic counters.


class _SimpleMetrics:
    """Thread-safe in-memory request/error/latency counters.

    Restart loss is acceptable per spec — these are for operational
    awareness, not billing or auditing.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_count = 0
        self._error_count = 0
        self._total_latency_ms = 0.0

    def record_request(self, latency_ms: float) -> None:
        with self._lock:
            self._request_count += 1
            self._total_latency_ms += latency_ms

    def record_error(self) -> None:
        with self._lock:
            self._error_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            avg_latency = (
                round(self._total_latency_ms / self._request_count, 2)
                if self._request_count > 0
                else 0.0
            )
            return {
                "request_count": self._request_count,
                "error_count": self._error_count,
                "avg_latency_ms": avg_latency,
            }


_metrics = _SimpleMetrics()


router = APIRouter(prefix="/v1", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=200,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def chat(
    request: ChatRequest,
    chain: RAGChain = Depends(get_rag_chain),
) -> ChatResponse:
    """
    Process a user query using RAG (Retrieval-Augmented Generation).

    This endpoint retrieves relevant documents based on the query, uses them as context,
    and generates a comprehensive answer via the LLM.

    Args:
        request: ChatRequest containing:
            - query: User's natural language question
            - top_k: Number of documents to retrieve (1-20)
            - include_sources: Whether to return source documents
            - temperature: LLM response creativity (0.0-1.0)
        chain: Injected RAGChain dependency for processing

    Returns:
        ChatResponse containing:
            - response: LLM-generated answer
            - query: Echo of the original query
            - sources: Retrieved documents (if requested)
            - execution_time_ms: Total processing time
            - model: LLM identifier used

    Raises:
        HTTPException (400): Validation error in request parameters
        HTTPException (500): Internal server error during processing
    """
    start_time = time.time()
    try:
        logger.info(f"Chat request: query={request.query[:50]}...")

        response = chain.invoke(
            query=request.query,
            top_k=request.top_k,
            temperature=request.temperature,
            include_sources=request.include_sources,
            latest_only=request.latest_only,
        )

        execution_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Chat response: time={execution_time_ms:.0f}ms, "
            f"sources={len(response.sources or [])}"
        )

        _metrics.record_request(execution_time_ms)

        return ChatResponse(
            response=response.response,
            query=request.query,
            sources=response.sources if request.include_sources else None,
            execution_time_ms=execution_time_ms,
            model=response.model,
            run_id=response.run_id,
        )

    except ValueError as e:
        logger.error(f"Chat validation error: {str(e)}")
        _metrics.record_error()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_request",
                "message": "Invalid request parameters",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        _metrics.record_error()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_error",
                "message": "Failed to process query",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


@router.post(
    "/feedback",
    status_code=200,
    responses={
        202: {"description": "Feedback accepted but LangSmith unreachable"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def feedback(
    request: FeedbackRequest,
    service=Depends(get_feedback_service),
) -> dict:
    """Record user feedback (like/dislike) correlated to a LangSmith run_id.

    Accepts feedback via the LangSmith Native Feedback API. If LangSmith is
    unreachable, the feedback is logged server-side and a 202 Accepted is
    returned instead of failing the request.

    Args:
        request: FeedbackRequest containing:
            - run_id: LangSmith run ID for feedback correlation
            - feedback_type: "like" (score=1.0) or "dislike" (score=0.0)
            - comment: Optional user comment (max 1000 chars)

    Returns:
        Dictionary with status "recorded" (200) or "accepted" (202)

    Raises:
        HTTPException (422): Validation error in request parameters
    """
    try:
        result = service.record_feedback(
            run_id=request.run_id,
            feedback_type=request.feedback_type,
            comment=request.comment,
        )
        if result.status == "accepted":
            return JSONResponse(
                content={"status": "accepted"},
                status_code=status.HTTP_202_ACCEPTED,
            )
        return {"status": "recorded"}
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_error",
                "message": "Failed to process feedback",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    status_code=200,
)
async def metrics() -> MetricsResponse:
    """
    Return lightweight operational counters since process startup.

    LangSmith handles the comprehensive observability dashboards;
    this endpoint provides simple request/error/latency counters
    that are health-check friendly.

    Returns:
        MetricsResponse with request_count, error_count, avg_latency_ms,
        and langsmith_dashboard_url (if tracing is configured).
    """
    data = _metrics.snapshot()

    langsmith_url = None
    if settings.enable_langsmith_tracing and settings.langsmith_api_key:
        # LangSmith dashboard URL pattern for a project
        project = settings.langsmith_project or "langchain-agent"
        langsmith_url = f"https://smith.langchain.com/o/default/projects/p/{project}"

    return MetricsResponse(
        request_count=data["request_count"],
        error_count=data["error_count"],
        avg_latency_ms=data["avg_latency_ms"],
        langsmith_dashboard_url=langsmith_url,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=200,
    responses={
        500: {"model": ErrorResponse, "description": "Health check failed"},
    },
)
async def health() -> HealthResponse:
    """
    Check service and database connectivity status.

    This endpoint performs a lightweight health check including:
    - Service availability
    - Database connectivity (Supabase/pgvector)
    - Overall system status

    Returns:
        HealthResponse containing:
            - status: "ok" if all checks pass, "error" if degraded
            - timestamp: Check timestamp
            - version: API version
            - db_connected: Database connectivity status

    Raises:
        Returns degraded status rather than raising, to allow clients to assess
        partial system health (e.g., API available but database down).
    """
    logger.debug("Health check requested")
    health_info = await check_health()

    return HealthResponse(
        status=health_info["status"],
        timestamp=datetime.utcnow(),
        db_connected=health_info.get("db_connected", False),
        llm_connected=health_info.get("llm_connected", False),
        embedding_connected=health_info.get("embedding_connected", False),
    )
