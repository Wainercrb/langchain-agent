"""LangSmith observability provider implementation.

Wraps LangSmith's tracing API (run trees, tags, metadata, feedback) behind
the ObservabilityProvider contract. If LangSmith is unreachable, errors are
caught and logged for graceful degradation.
"""

import uuid
from typing import Any, Callable, Dict, List, Literal, Optional

from langsmith import traceable
from langsmith import Client as LangSmithClient
from langsmith.run_trees import RunTree, _context as run_tree_context

from config import settings
from models.feedback import FeedbackResponse
from loggers import logger

from .base import ObservabilityProvider


class LangSmithObservabilityProvider(ObservabilityProvider):
    """LangSmith backend for tracing and feedback.

    Integrates with LangChain's @traceable decorator for automatic run
    lifecycle management, and uses langsmith.Client for feedback and
    health checks.
    """

    def __init__(self) -> None:
        """Initialize the LangSmith client."""
        self._client = LangSmithClient()

    def is_configured(self) -> bool:
        """Return True if LangSmith tracing is enabled with an API key."""
        return settings.enable_langsmith_tracing and bool(settings.langsmith_api_key)

    def get_current_run_id(self) -> Optional[str]:
        """Extract the current LangSmith run ID, or None if no trace is active."""
        current_run = run_tree_context.get_current_run_tree()
        return str(current_run.id) if current_run else None

    def apply_tags(self, run_id: str, tags: List[str]) -> None:
        """Attach tags to the active LangSmith run."""
        current_run = run_tree_context.get_current_run_tree()
        if current_run:
            current_run.add_tags(tags)

    def apply_metadata(self, run_id: str, metadata: Dict[str, Any]) -> None:
        """Attach metadata to the active LangSmith run.

        Silently skips if attachment fails — tracing should never break
        the primary request flow.
        """
        current_run = run_tree_context.get_current_run_tree()
        if not current_run:
            return
        try:
            current_run.add_metadata(metadata)
        except Exception as e:
            logger.debug(f"LangSmith metadata attachment failed: {e}")

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

        Args:
            run_id: LangSmith run ID for feedback correlation.
            feedback_type: "like" (score=1.0) or "dislike" (score=0.0).
            comment: Optional user comment (max 1000 chars).

        Returns:
            FeedbackResponse with status "recorded" on success or "accepted" on fallback.
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

    def dashboard_url(self) -> Optional[str]:
        """Return the LangSmith dashboard URL or None if not configured."""
        if not self.is_configured():
            return None
        project = settings.langsmith_project or "langchain-agent"
        return f"https://smith.langchain.com/o/default/projects/p/{project}"

    async def health_check(self) -> "CheckResult":
        """Verify LangSmith API is reachable using list_projects()."""
        from observability.health.checks import CheckResult

        if not self.is_configured():
            return CheckResult.success("LangSmith tracing not configured, skipping")
        try:
            list(self._client.list_projects(limit=1))
            return CheckResult.success("LangSmith API reachable")
        except Exception as e:
            return CheckResult.failure(f"LangSmith API unreachable: {str(e)}")

    def trace_call(
        self,
        fn: Callable,
        name: str,
        run_type: str,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Wrap the function call with LangSmith's @traceable decorator.

        This sets up the RunTree context so that get_current_run_id(),
        apply_tags(), and apply_metadata() work inside the function body.
        """
        wrapped = traceable(name=name, run_type=run_type)(fn)
        return wrapped(*args, **kwargs)
