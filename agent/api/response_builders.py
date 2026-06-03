"""Response builder helpers for API endpoints."""

from typing import Any, Optional

from models import ChatResponse, MetricsResponse
from models.observability.health import MonitoringStatusResponse
from models.observability.circuits import CircuitStatusResponse


def build_chat_response(
    response: Any,
    request_query: str,
    include_sources: bool,
    execution_time_ms: float,
) -> ChatResponse:
    """Build a ChatResponse from an agent invocation result."""

    def _safe_get(field: str, default: Any = None) -> Any:
        try:
            value = getattr(response, field, default)
            return value if isinstance(value, (dict, list, str, int, float, type(None))) else default
        except Exception:
            return default

    return ChatResponse(
        response=response.response,
        query=request_query,
        sources=response.sources if include_sources else None,
        execution_time_ms=execution_time_ms,
        model=response.model,
        run_id=_safe_get("run_id"),
        usage_metadata=_safe_get("usage_metadata"),
        llm_latency_ms=_safe_get("llm_latency_ms"),
        langsmith_tags=_safe_get("langsmith_tags"),
        agent_type=_safe_get("agent_type", "tool_calling"),
        tools_used=_safe_get("tools_used", []),
        chain_length=_safe_get("chain_length", 0),
        decision_quality=_safe_get("decision_quality", "suboptimal"),
        reasoning_summary=_safe_get("reasoning_summary"),
    )


def build_metrics_response(
    data: dict[str, Any],
    langsmith_dashboard_url: Optional[str] = None,
) -> MetricsResponse:
    """Build a MetricsResponse from a metrics snapshot."""
    return MetricsResponse(
        request_count=data["request_count"],
        error_count=data["error_count"],
        avg_latency_ms=data["avg_latency_ms"],
        total_input_tokens=data["total_input_tokens"],
        total_output_tokens=data["total_output_tokens"],
        avg_tokens_per_request=data["avg_tokens_per_request"],
        langsmith_dashboard_url=langsmith_dashboard_url,
        langsmith_audit_url=langsmith_dashboard_url,
        ai_decisions=data.get("ai_decisions"),
    )


def build_monitoring_response(
    enabled: bool,
    last_check: Any,
    interval_seconds: int,
    checks: list,
    overall_status: str,
) -> MonitoringStatusResponse:
    """Build a MonitoringStatusResponse."""
    return MonitoringStatusResponse(
        enabled=enabled,
        last_check=last_check,
        interval_seconds=interval_seconds,
        checks=checks,
        overall_status=overall_status,
    )


def build_circuit_response(circuits: dict[str, Any]) -> CircuitStatusResponse:
    """Build a CircuitStatusResponse."""
    return CircuitStatusResponse(circuits=circuits)
