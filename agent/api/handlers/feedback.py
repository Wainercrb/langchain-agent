"""Feedback endpoint — record user feedback correlated to LangSmith run_id."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from api.dependencies import get_feedback_service
from api.error_responses import internal_error_response
from infrastructure.logging import logger
from models import FeedbackRequest, ErrorResponse
from models.observability.decisions import DecisionLogEntry

router = APIRouter(prefix="/v1", tags=["feedback"])


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
