"""Response builder helpers for API endpoints."""

from typing import Any, Optional

from models import ChatResponse


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
        tracing_tags=_safe_get("tracing_tags"),
        agent_type=_safe_get("agent_type", "tool_calling"),
        tools_used=_safe_get("tools_used", []),
        chain_length=_safe_get("chain_length", 0),
        decision_quality=_safe_get("decision_quality", "suboptimal"),
        reasoning_summary=_safe_get("reasoning_summary"),
    )


