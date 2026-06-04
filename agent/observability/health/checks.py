"""Health checks — pure async functions, each returns ``CheckResult``.

Every check executes WITHOUT LLM invocation. Dependencies are passed as
parameters — no class, no constructor, no hidden state.
"""

import os
from typing import Awaitable, Callable, Optional

from langsmith import Client as LangSmithClient

from config import settings
from embeddings import GoogleEmbeddingsWrapper
from loggers import logger
from models.observability.decisions import DecisionQuality
from observability.base import CheckResult, ObservabilityProvider
from observability.decisions import DecisionTracker
from vector_store import VectorStore


# ── Helper (shared error isolation) ──────────────────────────────────


async def _run_and_catch(
    label: str,
    fn: Callable[[], Awaitable[CheckResult]],
) -> CheckResult:
    """Execute a check with error isolation.

    Args:
        label: Human-readable name for log messages.
        fn: Async callable that performs the check.

    Returns:
        ``CheckResult.success`` on success, ``CheckResult.failure`` on exception.
    """
    try:
        return await fn()
    except Exception as e:
        logger.error("Health check '%s' failed: %s", label, e)
        return CheckResult.failure(f"{label}: {e}")


# ── Individual checks (pure functions) ───────────────────────────────


async def check_database(vector_store: VectorStore) -> CheckResult:
    """Verify database connectivity via vector store health check."""

    async def _do() -> CheckResult:
        ok = await vector_store.health_check()
        if ok:
            return CheckResult.success("Database connection healthy")
        return CheckResult.failure("Database health check returned false")

    return await _run_and_catch("Database", _do)


async def check_observability_backend(
    observability: Optional[ObservabilityProvider],
) -> CheckResult:
    """Verify the configured observability backend is reachable."""
    if observability is None:
        return CheckResult.success("Observability not configured, skipping")
    return await observability.health_check()


async def check_embeddings_service(
    embeddings: GoogleEmbeddingsWrapper,
) -> CheckResult:
    """Verify embeddings provider is reachable."""

    async def _do() -> CheckResult:
        vec = embeddings.embed_query("health")
        if vec and len(vec) > 0:
            return CheckResult.success("Embeddings provider reachable")
        return CheckResult.failure("Embeddings returned empty vector")

    return await _run_and_catch("Embeddings", _do)


async def check_tracing_completeness(
    observability: Optional[ObservabilityProvider],
    metrics_store: Optional[Callable[[], dict]],
) -> CheckResult:
    """Compare LangSmith trace count against in-memory request count.

    Silently skips when observability or metrics store are not configured.
    """

    async def _do() -> CheckResult:
        if not observability or not observability.is_configured():
            return CheckResult.success("Observability not configured, skipping")
        if not metrics_store:
            return CheckResult.success("Metrics store not available, tracing check skipped")

        snapshot = metrics_store()
        request_count = snapshot.get("request_count", 0)
        if request_count == 0:
            return CheckResult.success("No requests made yet, tracing completeness N/A")

        runs = list(
            LangSmithClient().list_runs(
                project_name=settings.langsmith_project or "langchain-agent",
                limit=1000,
            )
        )
        trace_count = len(runs)

        if trace_count == 0:
            return CheckResult.failure(
                f"Tracing gap: {request_count} requests but 0 traces"
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

    return await _run_and_catch("Tracing completeness", _do)


async def check_process_memory() -> CheckResult:
    """Check process memory usage against configured threshold."""

    async def _do() -> CheckResult:
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

    return await _run_and_catch("Memory usage", _do)


async def check_decision_drift(
    decision_tracker: Optional[DecisionTracker],
) -> CheckResult:
    """Analyze recent decision quality trends and alert on degradation."""

    async def _do() -> CheckResult:
        if decision_tracker is None:
            return CheckResult.success("No decision tracker available, drift check skipped")

        window_size = 50
        quality_threshold = 0.5

        all_decisions = decision_tracker.get_recent(window_size)
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

    return await _run_and_catch("Decision drift", _do)
