"""Health verifier — individual check methods for automated monitoring.

All checks executes WITHOUT LLM invocation. Each returns a ``CheckResult``
with status and human-readable detail.
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional

from config import settings
from embeddings import GoogleEmbeddingsWrapper
from loggers import logger
from observability.decisions import DecisionTracker
from agent.observability.base import ObservabilityProvider
from vector_store import VectorStore


@dataclass(frozen=True)
class CheckResult:
    """Immutable result of a single health check.

    Attributes:
        ok: Whether the check passed.
        detail: Human-readable explanation of the result.
    """

    ok: bool
    detail: str

    @classmethod
    def success(cls, detail: str) -> "CheckResult":
        """Create a passing check result."""
        return cls(ok=True, detail=detail)

    @classmethod
    def failure(cls, detail: str) -> "CheckResult":
        """Create a failing check result."""
        return cls(ok=False, detail=detail)


class HealthVerifier:
    """Performs individual health checks without LLM invocation.

    Each check method returns a ``CheckResult`` with status and detail.
    Checks that depend on external services perform real connectivity tests.

    Args:
        vector_store: Vector store instance for DB connectivity checks.
        embeddings: Embeddings provider instance for embedding checks.
        metrics_store: Callable returning request metrics snapshot.
        decision_tracker: DecisionTracker for drift analysis.
        observability: ObservabilityProvider for backend checks.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embeddings: GoogleEmbeddingsWrapper,
        metrics_store: Optional[Callable[[], dict]] = None,
        decision_tracker: Optional[DecisionTracker] = None,
        observability: Optional[ObservabilityProvider] = None,
    ) -> None:
        self._vector_store = vector_store
        self._embeddings = embeddings
        self._metrics_store = metrics_store
        self._decision_tracker = decision_tracker
        self._observability = observability

    async def check_db(self) -> CheckResult:
        """Verify database connectivity via vector store health check."""
        try:
            result = await self._vector_store.health_check()
            if result:
                return CheckResult.success("Database connection healthy")
            return CheckResult.failure("Database health check returned false")
        except Exception as e:
            return CheckResult.failure(f"Database connection failed: {str(e)}")

    async def check_observability(self) -> CheckResult:
        """Verify the configured observability backend is reachable."""
        if self._observability is None:
            return CheckResult.success("Observability not configured, skipping")
        return await self._observability.health_check()

    async def check_embeddings(self) -> CheckResult:
        """Verify embeddings provider is reachable."""
        try:
            vec = self._embeddings.embed_query("health")
            if vec and len(vec) > 0:
                return CheckResult.success("Embeddings provider reachable")
            return CheckResult.failure("Embeddings returned empty vector")
        except Exception as e:
            return CheckResult.failure(f"Embeddings provider unreachable: {str(e)}")

    async def check_tracing_completeness(self) -> CheckResult:
        """Compare trace count against in-memory request count.

        Delegates to the configured observability provider for backend-specific
        trace counting. Falls back to a skip if the provider doesn't support
        trace enumeration.
        """
        if self._observability is None or not self._observability.is_configured():
            return CheckResult.success("Observability not configured, skipping")

        if not self._metrics_store:
            return CheckResult.success("Metrics store not available, tracing check skipped")

        try:
            snapshot = self._metrics_store()
            request_count = snapshot.get("request_count", 0)

            if request_count == 0:
                return CheckResult.success("No requests made yet, tracing completeness N/A")

            # LangSmith-specific trace counting
            from langsmith import Client as LangSmithClient

            client = LangSmithClient()
            window = settings.monitoring_tracing_window_seconds

            runs = list(
                client.list_runs(
                    project_name=settings.langsmith_project or "langchain-agent",
                    limit=1000,
                )
            )
            trace_count = len(runs)

            if trace_count == 0 and request_count > 0:
                return CheckResult.failure(
                    f"Tracing gap: {request_count} requests but 0 traces "
                    f"in last {window}s window"
                )

            mismatch = abs(request_count - trace_count)
            if mismatch > max(1, request_count * 0.1):
                return CheckResult.failure(
                    f"Trace mismatch: {request_count} requests vs {trace_count} "
                    f"traces ({mismatch} difference)"
                )

            return CheckResult.success(
                f"Tracing complete: {trace_count} traces for {request_count} requests"
            )
        except ImportError:
            return CheckResult.success("LangSmith client not available, tracing check skipped")
        except Exception as e:
            return CheckResult.failure(f"Tracing completeness check failed: {str(e)}")

    async def check_memory_usage(self) -> CheckResult:
        """Check process memory usage against threshold.

        Uses psutil if available, otherwise skips gracefully.
        """
        try:
            import psutil

            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            threshold = settings.monitoring_memory_threshold_mb

            if memory_mb > threshold:
                return CheckResult.failure(
                    f"Memory usage {memory_mb:.1f}MB exceeds threshold {threshold}MB"
                )
            return CheckResult.success(
                f"Memory usage {memory_mb:.1f}MB (threshold: {threshold}MB)"
            )
        except ImportError:
            return CheckResult.success("psutil not available, memory check skipped")
        except Exception as e:
            return CheckResult.failure(f"Memory check failed: {str(e)}")

    async def check_decision_drift(self) -> CheckResult:
        """Analyze recent decision quality trends and alert on degradation."""
        if self._decision_tracker is None:
            return CheckResult.success("No decision tracker available, drift check skipped")

        try:
            from models.observability.decisions import DecisionQuality

            window_size = 50
            quality_threshold = 0.5

            all_decisions = self._decision_tracker.get_recent(window_size)
            if len(all_decisions) < 10:
                return CheckResult.success(
                    f"Insufficient data for drift analysis: "
                    f"{len(all_decisions)} decisions (need >= 10)"
                )

            optimal_count = sum(
                1 for d in all_decisions
                if d.decision_quality == DecisionQuality.OPTIMAL
            )
            poor_count = sum(
                1 for d in all_decisions
                if d.decision_quality == DecisionQuality.POOR
            )
            optimal_ratio = optimal_count / len(all_decisions)
            poor_ratio = poor_count / len(all_decisions)

            if optimal_ratio < quality_threshold:
                return CheckResult.failure(
                    f"Decision quality degradation: "
                    f"{optimal_ratio:.0%} optimal in last {len(all_decisions)} decisions "
                    f"(threshold: {quality_threshold:.0%}), "
                    f"{poor_ratio:.0%} poor"
                )

            return CheckResult.success(
                f"Decision quality stable: "
                f"{optimal_ratio:.0%} optimal, {poor_ratio:.0%} poor "
                f"(last {len(all_decisions)} decisions)"
            )
        except Exception as e:
            return CheckResult.failure(f"Decision drift check failed: {str(e)}")
