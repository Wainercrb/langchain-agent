"""No-op observability provider — safe fallback for local dev or when tracing is disabled.

Returns UUIDs for run IDs, silently ignores tags/metadata/feedback,
and reports healthy. This ensures the app works without any external
observability backend configured.
"""

import uuid
from typing import Any, Dict, List, Literal, Optional

from models.feedback import FeedbackResponse

from .provider import ObservabilityProvider


class NoOpObservabilityProvider(ObservabilityProvider):
    """No-op backend that satisfies the ObservabilityProvider contract.

    Used when no observability backend is configured. All operations are
    safe no-ops — the app functions normally without tracing.
    """

    def is_configured(self) -> bool:
        """No-op is never 'configured' — it's the fallback."""
        return False

    def get_current_run_id(self) -> Optional[str]:
        """Return a fresh UUID — no active trace exists."""
        return str(uuid.uuid4())

    def apply_tags(self, run_id: str, tags: List[str]) -> None:
        """No-op — tags are discarded."""

    def apply_metadata(self, run_id: str, metadata: Dict[str, Any]) -> None:
        """No-op — metadata is discarded."""

    def record_feedback(
        self,
        run_id: str,
        feedback_type: Literal["like", "dislike"],
        comment: Optional[str] = None,
    ) -> FeedbackResponse:
        """Accept feedback without persisting it."""
        return FeedbackResponse(status="accepted")

    def dashboard_url(self) -> Optional[str]:
        """No dashboard available."""
        return None

    async def health_check(self) -> "CheckResult":
        """Always healthy — no external dependency."""
        from observability.health.checks import CheckResult

        return CheckResult.success("Observability disabled (no-op)")
