"""Abstract feedback provider interface.

Strategy Pattern: swap LangSmith ↔ Supabase ↔ No-op by changing
the concrete class wired in services/container.py.
"""

from abc import ABC, abstractmethod
from typing import Literal, Optional

from models.feedback import FeedbackResponse


class FeedbackProvider(ABC):
    """Abstract base for pluggable feedback backends.

    Records user feedback (like/dislike) correlated to a run ID.
    Implementations decide where the feedback is persisted.
    """

    @abstractmethod
    def record_feedback(
        self,
        run_id: str,
        feedback_type: Literal["like", "dislike"],
        comment: Optional[str] = None,
    ) -> FeedbackResponse:
        """Record user feedback.

        Args:
            run_id: Unique identifier for the run/session being rated
            feedback_type: "like" (score=1.0) or "dislike" (score=0.0)
            comment: Optional free-text comment from the user

        Returns:
            FeedbackResponse with status "recorded" on success or "accepted"
            if the backend is unreachable (graceful degradation).
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
