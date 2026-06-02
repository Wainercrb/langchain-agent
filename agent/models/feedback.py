"""Feedback models for Like/Dislike user feedback via LangSmith."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    """Request model for POST /v1/feedback endpoint.

    Captures user feedback (like/dislike) correlated to a LangSmith run_id.

    Attributes:
        run_id: LangSmith run ID for feedback correlation (required)
        feedback_type: Type of feedback — "like" (score=1.0) or "dislike" (score=0.0)
        comment: Optional user comment providing additional context (max 1000 chars)
    """

    run_id: str = Field(
        ...,
        description="LangSmith run ID for feedback correlation",
    )
    feedback_type: Literal["like", "dislike"] = Field(
        ...,
        description="Feedback type: 'like' (score=1.0) or 'dislike' (score=0.0)",
    )
    comment: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional user comment (max 1000 characters)",
    )


class FeedbackResponse(BaseModel):
    """Response model for POST /v1/feedback endpoint.

    Attributes:
        status: "recorded" when feedback was stored in LangSmith,
                "accepted" when LangSmith was unreachable but feedback was logged server-side
    """

    status: Literal["recorded", "accepted"] = Field(
        ...,
        description="'recorded' when persisted to LangSmith, 'accepted' when logged server-side only",
    )
