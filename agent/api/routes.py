import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from api.dependencies import check_health, get_agent, get_decision_tracker, get_feedback_service
from api.error_responses import internal_error_response, not_found_response, validation_error_response
from api.metrics import build_metrics_snapshot, get_llm_usage_metrics, get_request_metrics
from api.response_builders import (
    build_chat_response,
    build_circuit_response,
    build_metrics_response,
    build_monitoring_response,
)
from config import settings
from models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    FeedbackRequest,
    HealthResponse,
    MetricsResponse,
    MonitoringStatusResponse,
    CircuitStatusResponse,
)
from models.observability.decisions import DecisionLogEntry, DecisionMetricsResponse
from infrastructure.logging import logger
from utils.exceptions import Severity, AllProvidersExhaustedError

router = APIRouter(prefix="/v1", tags=["chat"])


def _extract_tokens(response: Any) -> Tuple[int, int]:
    """Extract input/output token counts from a ChatResponse.

    Returns (0, 0) if token metadata is unavailable rather than raising.
    """
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        return 0, 0

    return (
        usage.get("input_tokens", usage.get("prompt_tokens", 0)),
        usage.get("output_tokens", usage.get("completion_tokens", 0)),
    )


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
    processor=Depends(get_agent),
) -> ChatResponse:
    """
    Process a user query using the configured agent strategy.

    The container decides whether to use ToolCallingAgent (intelligent tool
    selection) or RAGChainAgent (legacy always-retrieve). The endpoint is
    agnostic — it just calls agent.invoke().

    Args:
        request: ChatRequest containing:
            - query: User's natural language question
            - top_k: Number of documents to retrieve (1-20)
            - include_sources: Whether to return source documents
            - temperature: LLM response creativity (0.0-1.0)
        processor: Injected Agent from container.

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
    from infrastructure.container import alert_service

    start_time = time.time()
    try:
        logger.info(f"Chat request: query={request.query[:50]}...")

        response = processor.invoke(
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

        get_request_metrics().record_request(execution_time_ms)

        input_tokens, output_tokens = _extract_tokens(response)
        get_llm_usage_metrics().record_tokens(input_tokens, output_tokens)

        return build_chat_response(
            response=response,
            request_query=request.query,
            include_sources=request.include_sources,
            execution_time_ms=execution_time_ms,
        )

    except ValueError as e:
        logger.error(f"Chat validation error: {str(e)}")
        get_request_metrics().record_error()
        await alert_service.send_alert(
            severity=Severity.WARNING,
            message=f"Chat validation error: {str(e)[:100]}",
            error=e,
            metadata={"path": "/v1/chat", "query": request.query[:50]},
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=validation_error_response("Invalid request parameters"),
        )

    except AllProvidersExhaustedError as e:
        logger.error(f"All LLM providers exhausted: {str(e)}")
        get_request_metrics().record_error()
        await alert_service.send_alert(
            severity=Severity.CRITICAL,
            message="All LLM providers exhausted — service degraded",
            error=e,
            metadata={"path": "/v1/chat", "query": request.query[:50]},
        )

        execution_time_ms = (time.time() - start_time) * 1000
        return ChatResponse(
            response=(
                "I'm temporarily unable to process your request because all AI providers "
                "are unavailable. Please try again in a few minutes."
            ),
            query=request.query,
            sources=None,
            execution_time_ms=execution_time_ms,
            model="unavailable",
            run_id=str(uuid.uuid4()),
            usage_metadata=None,
            llm_latency_ms=0,
            langsmith_tags=["degraded:true", "reason:all_providers_exhausted"],
        )

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        get_request_metrics().record_error()
        await alert_service.send_alert(
            severity=Severity.ERROR,
            message=f"Chat processing failed: {str(e)[:200]}",
            error=e,
            metadata={"path": "/v1/chat", "query": request.query[:50]},
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=internal_error_response("Failed to process query"),
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

    Feedback is also correlated with the in-memory DecisionTracker so that
    decision quality analysis can incorporate explicit user signals.

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
    from infrastructure.container import decision_tracker

    # Correlate feedback with DecisionTracker
    feedback_payload = {
        "type": request.feedback_type,
        "comment": request.comment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    existing = decision_tracker.get_by_run_id(request.run_id)
    if existing:
        updated = DecisionLogEntry(
            **{**existing.model_dump(), "user_feedback": feedback_payload},
        )
        decision_tracker.record(updated)
        logger.info(
            f"Feedback correlated with decision: run_id={request.run_id}, "
            f"type={request.feedback_type}"
        )

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
            detail=internal_error_response("Failed to process feedback"),
        )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    status_code=200,
)
async def metrics(
    tracker=Depends(get_decision_tracker),
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
    data = build_metrics_snapshot(decision_tracker=tracker)

    langsmith_url = None
    if settings.enable_langsmith_tracing and settings.langsmith_api_key:
        project = settings.langsmith_project or "langchain-agent"
        langsmith_url = f"https://smith.langchain.com/o/default/projects/p/{project}"

    return build_metrics_response(data=data, langsmith_dashboard_url=langsmith_url)


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


@router.get(
    "/decisions",
    response_model=DecisionMetricsResponse,
    status_code=200,
)
async def list_decisions(
    from_date: Optional[str] = Query(None, alias="from", description="ISO 8601 start date filter"),
    to_date: Optional[str] = Query(None, alias="to", description="ISO 8601 end date filter"),
    tool: Optional[str] = Query(None, description="Filter by tool name"),
    quality: Optional[str] = Query(None, description="Filter by decision quality"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Results per page"),
    tracker=Depends(get_decision_tracker),
) -> DecisionMetricsResponse:
    """Query AI decision records with optional filters and pagination.

    Args:
        from_date: ISO 8601 start date (inclusive).
        to_date: ISO 8601 end date (inclusive).
        tool: Filter by tool name.
        quality: Filter by decision_quality (optimal, suboptimal, poor).
        page: Page number (1-indexed).
        per_page: Results per page (max 100).
        tracker: Injected DecisionTracker.

    Returns:
        DecisionMetricsResponse with filtered decisions and aggregates.
    """
    return tracker.query(
        from_date=from_date,
        to_date=to_date,
        tool=tool,
        quality=quality,
        page=page,
        per_page=per_page,
    )


@router.get(
    "/decisions/{run_id}",
    status_code=200,
    responses={
        404: {"model": ErrorResponse, "description": "Decision not found"},
    },
)
async def get_decision(
    run_id: str,
    tracker=Depends(get_decision_tracker),
) -> DecisionLogEntry:
    """Retrieve a single decision record by run_id.

    Args:
        run_id: LangSmith run ID to look up.
        tracker: Injected DecisionTracker.

    Returns:
        Full DecisionLogEntry with tool chain and metadata.

    Raises:
        HTTPException (404): If no decision record exists for the run_id.
    """
    entry = tracker.get_by_run_id(run_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=not_found_response("Decision", f"run_id: {run_id}"),
        )
    return entry


@router.get(
    "/monitoring/status",
    response_model=MonitoringStatusResponse,
    status_code=200,
)
async def monitoring_status() -> MonitoringStatusResponse:
    """Return the last check results for each monitoring verification task.

    Returns:
        MonitoringStatusResponse containing:
            - enabled: Whether monitoring is enabled
            - last_check: Timestamp of last complete check cycle
            - interval_seconds: Configured check interval
            - checks: List of individual check results
            - overall_status: "ok", "degraded", or "error"
    """
    from infrastructure.container import _monitoring_scheduler

    results = _monitoring_scheduler.last_results
    checks = list(results.values())

    return build_monitoring_response(
        enabled=settings.monitoring_enabled,
        last_check=_monitoring_scheduler.last_check,
        interval_seconds=settings.monitoring_interval_seconds,
        checks=checks,
        overall_status=_monitoring_scheduler.overall_status,
    )


@router.get(
    "/llm/circuits",
    response_model=CircuitStatusResponse,
    status_code=200,
)
async def circuit_status() -> CircuitStatusResponse:
    """Return current circuit breaker status for all LLM providers.

    Returns:
        CircuitStatusResponse with provider name -> circuit state mapping.
    """
    from infrastructure.container import llm

    return build_circuit_response(
        circuits=llm.circuit_status() if hasattr(llm, "circuit_status") else {},
    )
