"""Feedback service wrapping LangSmith Native Feedback API.

Isolates API routes from LangSmith SDK details. Follows the same
container-friendly pattern as other services in agent/services/.
"""

from typing import Literal, Optional

from langsmith import Client as LangSmithClient

from models.feedback import FeedbackResponse
from services.logging import Console

logger = Console()


class FeedbackService:
    """Service for recording user feedback via LangSmith.

    Encapsulates langsmith.Client().create_feedback() behind a simple
    record_feedback() interface. Returns a FeedbackResponse that the
    API layer maps to the appropriate HTTP status code.

    Attributes:
        _client: LangSmith API client instance
    """

    def __init__(self) -> None:
        """Initialize the LangSmith client."""
        self._client = LangSmithClient()

    def record_feedback(
        self,
        run_id: str,
        feedback_type: Literal["like", "dislike"],
        comment: Optional[str] = None,
    ) -> FeedbackResponse:
        """Record user feedback in LangSmith.

        Maps feedback_type to a numeric score:
            - "like"    → score=1.0
            - "dislike" → score=0.0

        If LangSmith is unreachable or raises an exception, the error is
        caught and logged server-side, returning status="accepted".

        Args:
            run_id: LangSmith run ID to associate feedback with
            feedback_type: "like" (score=1.0) or "dislike" (score=0.0)
            comment: Optional user comment (max 1000 chars per FeedbackRequest)

        Returns:
            FeedbackResponse with status "recorded" on success or "accepted" on fallback
        """
        score = 1.0 if feedback_type == "like" else 0.0

        try:
            self._client.create_feedback(
                run_id=run_id,
                key="user-feedback",
                score=score,
                comment=comment,
            )
            logger.info(
                f"Feedback recorded: run_id={run_id}, "
                f"feedback_type={feedback_type}, score={score}"
            )
            return FeedbackResponse(status="recorded")
        except Exception as e:
            logger.warning(
                f"Failed to record feedback in LangSmith: run_id={run_id}, "
                f"feedback_type={feedback_type}, error={str(e)}"
            )
            return FeedbackResponse(status="accepted")
