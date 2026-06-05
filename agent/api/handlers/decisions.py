"""Decision endpoints — query and retrieve AI decision records."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.api_errors import not_found_response
from container import decision_tracker
from models import ErrorResponse
from models.observability.decisions import DecisionLogEntry, DecisionMetricsResponse

router = APIRouter(prefix="/v1", tags=["decisions"])


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
    tracker=Depends(lambda: decision_tracker),
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
    tracker=Depends(lambda: decision_tracker),
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
