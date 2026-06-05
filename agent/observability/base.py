"""Observability provider interface — swap backends without touching agent code.

Strategy Pattern: LangSmith ↔ future backends (OpenTelemetry, Datadog, etc.)
by changing the concrete class wired in ``container.py``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional

from models.feedback import FeedbackResponse


# ── Shared result type (belongs here, not in health/checks.py) ──────────


@dataclass(frozen=True)
class CheckResult:
    """Immutable result of a single health check."""

    ok: bool
    detail: str

    @classmethod
    def success(cls, detail: str) -> "CheckResult":
        return cls(ok=True, detail=detail)

    @classmethod
    def failure(cls, detail: str) -> "CheckResult":
        return cls(ok=False, detail=detail)


# ── ABC ────────────────────────────────────────────────────────────────


class ObservabilityProvider(ABC):
    """Pluggable observability backend (tracing, feedback, health checks)."""

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
        """Record user feedback correlated to a run ID."""

    # ── Discovery & health ───────────────────────────────────────────

    @abstractmethod
    def dashboard_url(self) -> Optional[str]:
        """Return a link to the observability dashboard, or None."""

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if this backend has valid credentials/config."""

    @abstractmethod
    async def health_check(self) -> CheckResult:
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
        integrate with their backend's trace lifecycle.
        """
        return fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ── Global accessor ───────────────────────────────────────────────────

_provider: Optional[ObservabilityProvider] = None


def set_observability_provider(provider: ObservabilityProvider) -> None:
    """Set the global observability provider singleton."""
    global _provider
    _provider = provider


def get_observability_provider() -> Optional[ObservabilityProvider]:
    """Return the configured provider, or None if unset."""
    return _provider
