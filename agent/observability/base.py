"""Abstract observability provider interface.

Strategy Pattern: swap LangSmith ↔ No-Op ↔ future backends (OpenTelemetry,
Datadog, etc.) by changing the concrete class wired in container.py.

Unifies tracing (run lifecycle, tags, metadata) and feedback (like/dislike)
into a single injection point.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional

from models.feedback import FeedbackResponse

if TYPE_CHECKING:
    from observability.health.checks import CheckResult


class ObservabilityProvider(ABC):
    """Abstract base for pluggable observability backends.

    Handles trace lifecycle (start/end, tags, metadata), user feedback,
    and health checks. Implementations decide where data is persisted.
    """

    # ── Trace lifecycle ──────────────────────────────────────────────

    @abstractmethod
    def get_current_run_id(self) -> Optional[str]:
        """Return the active run ID, or None if no trace is in progress."""

    @abstractmethod
    def apply_tags(self, run_id: str, tags: List[str]) -> None:
        """Attach tags to an active trace."""

    @abstractmethod
    def apply_metadata(self, run_id: str, metadata: Dict[str, Any]) -> None:
        """Attach key-value metadata to an active trace."""

    # ── Feedback ─────────────────────────────────────────────────────

    @abstractmethod
    def record_feedback(
        self,
        run_id: str,
        feedback_type: Literal["like", "dislike"],
        comment: Optional[str] = None,
    ) -> FeedbackResponse:
        """Record user feedback correlated to a run ID.

        Args:
            run_id: Unique identifier for the run being rated.
            feedback_type: "like" (score=1.0) or "dislike" (score=0.0).
            comment: Optional free-text comment from the user.

        Returns:
            FeedbackResponse with status "recorded" or "accepted".
        """

    # ── Discovery & health ───────────────────────────────────────────

    @abstractmethod
    def dashboard_url(self) -> Optional[str]:
        """Return a link to the observability dashboard, or None."""

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if this backend has valid credentials/config."""

    @abstractmethod
    async def health_check(self) -> "CheckResult":
        """Verify the observability backend is reachable."""

    # ── Decorator support ────────────────────────────────────────────

    def trace_call(
        self,
        fn: Callable,
        name: str,
        run_type: str,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Wrap a function call with tracing.

        Default: call the function directly. Subclasses override to
        integrate with their backend's trace lifecycle (e.g. LangSmith's
        @traceable decorator).

        Args:
            fn: The function to wrap.
            name: Human-readable name for the trace span.
            run_type: Type of run ("chain", "llm", "tool", etc.).
            args: Positional arguments to pass to fn.
            kwargs: Keyword arguments to pass to fn.

        Returns:
            The return value of fn.
        """
        return fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ── Global accessor (set by container.py at startup) ─────────────────

_provider: Optional[ObservabilityProvider] = None


def set_observability_provider(provider: ObservabilityProvider) -> None:
    """Set the global observability provider singleton."""
    global _provider
    _provider = provider


def get_observability_provider() -> ObservabilityProvider:
    """Return the configured provider, falling back to NoOp if unset."""
    return _provider
