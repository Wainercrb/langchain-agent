"""Health verifier — individual check methods for automated monitoring.

All checks execute WITHOUT LLM invocation. Each returns (ok: bool, detail: str).
"""

import os
from datetime import datetime, timezone
from typing import Tuple

from config import settings
from infrastructure.logging import logger


class HealthVerifier:
    """Performs individual health checks without LLM invocation."""

    def __init__(
        self,
        vector_store=None,
        embeddings=None,
    ) -> None:
        self._vector_store = vector_store
        self._embeddings = embeddings

    async def check_db(self) -> Tuple[bool, str]:
        """Verify database connectivity via vector store health check."""
        try:
            result = await self._vector_store.health_check()
            if result:
                return True, "Database connection healthy"
            return False, "Database health check returned false"
        except Exception as e:
            return False, f"Database connection failed: {str(e)}"

    async def check_langsmith(self) -> Tuple[bool, str]:
        """Verify LangSmith API is reachable using list_projects()."""
        if not settings.enable_langsmith_tracing or not settings.langsmith_api_key:
            return True, "LangSmith tracing not configured, skipping"
        try:
            from langsmith import Client as LangSmithClient
            client = LangSmithClient()
            list(client.list_projects(limit=1))
            return True, "LangSmith API reachable"
        except Exception as e:
            return False, f"LangSmith API unreachable: {str(e)}"

    async def check_embeddings(self) -> Tuple[bool, str]:
        """Verify embeddings provider is reachable."""
        try:
            vec = self._embeddings.embed_query("health")
            if vec and len(vec) > 0:
                return True, "Embeddings provider reachable"
            return False, "Embeddings returned empty vector"
        except Exception as e:
            return False, f"Embeddings provider unreachable: {str(e)}"

    async def check_tracing_completeness(self) -> Tuple[bool, str]:
        """Compare LangSmith trace count against in-memory request count.

        Uses langsmith.Client.list_runs() to count recent traces and compares
        against RequestMetrics.request_count over a configurable time window.
        """
        if not settings.enable_langsmith_tracing or not settings.langsmith_api_key:
            return True, "LangSmith tracing not configured, skipping"
        try:
            from langsmith import Client as LangSmithClient
            from api.metrics.request import get_request_metrics

            client = LangSmithClient()
            now = datetime.now(timezone.utc)
            window = settings.monitoring_tracing_window_seconds

            runs = list(
                client.list_runs(
                    project_name=settings.langsmith_project or "langchain-agent",
                    start_time=now.timestamp() - window,
                    limit=1000,
                )
            )
            trace_count = len(runs)
            snapshot = get_request_metrics().snapshot()
            request_count = snapshot.get("request_count", 0)

            if request_count == 0:
                return True, "No requests made yet, tracing completeness N/A"

            if trace_count == 0 and request_count > 0:
                return False, (
                    f"Tracing gap: {request_count} requests but 0 traces "
                    f"in last {window}s window"
                )

            mismatch = abs(request_count - trace_count)
            if mismatch > max(1, request_count * 0.1):
                return False, (
                    f"Trace mismatch: {request_count} requests vs {trace_count} "
                    f"traces ({mismatch} difference)"
                )

            return True, f"Tracing complete: {trace_count} traces for {request_count} requests"
        except Exception as e:
            return False, f"Tracing completeness check failed: {str(e)}"

    async def check_memory_usage(self) -> Tuple[bool, str]:
        """Check process memory usage against threshold.

        Uses psutil if available, otherwise skips gracefully.
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            threshold = settings.monitoring_memory_threshold_mb

            if memory_mb > threshold:
                return False, (
                    f"Memory usage {memory_mb:.1f}MB exceeds threshold {threshold}MB"
                )
            return True, f"Memory usage {memory_mb:.1f}MB (threshold: {threshold}MB)"
        except ImportError:
            return True, "psutil not available, memory check skipped"
        except Exception as e:
            return False, f"Memory check failed: {str(e)}"

    async def check_log_rotation(self) -> Tuple[bool, str]:
        """Always passes — log rotation is managed by CloudWatch retention policies."""
        return True, "Log rotation managed by CloudWatch retention policies"

    async def check_decision_drift(self, decision_tracker=None) -> Tuple[bool, str]:
        """Analyze recent decision quality trends and alert on degradation.

        Compares the optimal decision ratio in the last N decisions against
        a threshold. If the ratio drops below the threshold, it signals
        potential model drift, prompt degradation, or tool availability issues.

        Args:
            decision_tracker: DecisionTracker instance to query.

        Returns:
            Tuple of (ok, detail).
        """
        if decision_tracker is None:
            return True, "No decision tracker available, drift check skipped"

        try:
            from models.observability.decisions import DecisionQuality

            # Analyze the last 50 decisions for drift
            window_size = 50
            quality_threshold = 0.5  # 50% optimal ratio minimum

            all_decisions = list(decision_tracker._store)[-window_size:]
            if len(all_decisions) < 10:
                return True, (
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
                return False, (
                    f"Decision quality degradation: "
                    f"{optimal_ratio:.0%} optimal in last {len(all_decisions)} decisions "
                    f"(threshold: {quality_threshold:.0%}), "
                    f"{poor_ratio:.0%} poor"
                )

            return True, (
                f"Decision quality stable: "
                f"{optimal_ratio:.0%} optimal, {poor_ratio:.0%} poor "
                f"(last {len(all_decisions)} decisions)"
            )
        except Exception as e:
            return False, f"Decision drift check failed: {str(e)}"
