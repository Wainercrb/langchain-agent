"""Chat endpoint — process user queries via the configured agent strategy."""

import json
import time
import uuid
from datetime import datetime
from typing import Any, Tuple

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.api_errors import internal_error_response, validation_error_response
from api.api_responses import build_chat_response
from container import agent
from loggers import logger
from api.metrics_store import get_llm_usage_metrics, get_request_metrics
from models import ChatRequest, ChatResponse, ErrorResponse
from shared.exceptions import Severity, AllProvidersExhaustedError


class _Encoder(json.JSONEncoder):
    """JSON encoder that handles Pydantic models and datetime objects."""

    def default(self, obj: object) -> object:
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

router = APIRouter(prefix="/v1", tags=["chat"])


def _extract_tokens(response: Any) -> Tuple[int, int]:
    """Extract input/output token counts from a ChatResponse.

    Returns (0, 0) if token metadata is unavailable rather than raising.
    """
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        return 0, 0

    return (
        usage.get("input_tokens", usage.get("prompt_tokens", 0)),
        usage.get("output_tokens", usage.get("completion_tokens", 0)),
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=200,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def chat(
    request: ChatRequest,
    processor=Depends(lambda: agent),
) -> ChatResponse:
    """
    Process a user query using the configured agent strategy.

    The container decides whether to use ToolCallingAgent (intelligent tool
    selection) or RAGChainAgent (legacy always-retrieve). The endpoint is
    agnostic — it just calls agent.invoke().

    Args:
        request: ChatRequest containing:
            - query: User's natural language question
            - top_k: Number of documents to retrieve (1-20)
            - include_sources: Whether to return source documents
            - temperature: LLM response creativity (0.0-1.0)
        processor: Injected Agent from container.

    Returns:
        ChatResponse containing:
            - response: LLM-generated answer
            - query: Echo of the original query
            - sources: Retrieved documents (if requested)
            - execution_time_ms: Total processing time
            - model: LLM identifier used

    Raises:
        HTTPException (400): Validation error in request parameters
        HTTPException (500): Internal server error during processing
    """
    from container import alert_service

    start_time = time.time()
    try:
        logger.info(f"Chat request: query={request.query[:50]}...")

        response = processor.invoke(
            query=request.query,
            top_k=request.top_k,
            temperature=request.temperature,
            include_sources=request.include_sources,
        )

        execution_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Chat response: time={execution_time_ms:.0f}ms, "
            f"sources={len(response.sources or [])}"
        )

        get_request_metrics().record_request(execution_time_ms)

        input_tokens, output_tokens = _extract_tokens(response)
        get_llm_usage_metrics().record_tokens(input_tokens, output_tokens)

        return build_chat_response(
            response=response,
            request_query=request.query,
            include_sources=request.include_sources,
            execution_time_ms=execution_time_ms,
        )

    except ValueError as e:
        logger.error(f"Chat validation error: {str(e)}")
        get_request_metrics().record_error()
        await alert_service.send_alert(
            severity=Severity.WARNING,
            message=f"Chat validation error: {str(e)[:100]}",
            error=e,
            metadata={"path": "/v1/chat", "query": request.query[:50]},
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=validation_error_response("Invalid request parameters"),
        )

    except AllProvidersExhaustedError as e:
        logger.error(f"All LLM providers exhausted: {str(e)}")
        get_request_metrics().record_error()
        await alert_service.send_alert(
            severity=Severity.CRITICAL,
            message="All LLM providers exhausted — service degraded",
            error=e,
            metadata={"path": "/v1/chat", "query": request.query[:50]},
        )

        execution_time_ms = (time.time() - start_time) * 1000
        return ChatResponse(
            response=(
                "I'm temporarily unable to process your request because all AI providers "
                "are unavailable. Please try again in a few minutes."
            ),
            query=request.query,
            sources=None,
            execution_time_ms=execution_time_ms,
            model="unavailable",
            run_id=str(uuid.uuid4()),
            usage_metadata=None,
            llm_latency_ms=0,
            tracing_tags=["degraded:true", "reason:all_providers_exhausted"],
        )

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        get_request_metrics().record_error()
        await alert_service.send_alert(
            severity=Severity.ERROR,
            message=f"Chat processing failed: {str(e)[:200]}",
            error=e,
            metadata={"path": "/v1/chat", "query": request.query[:50]},
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=internal_error_response("Failed to process query"),
        )


# ── Streaming endpoint ────────────────────────────────────────────────


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    processor=Depends(lambda: agent),
) -> StreamingResponse:
    """Process a user query and stream results via Server-Sent Events.

    Same input schema as ``POST /v1/chat`` but returns an SSE stream
    instead of a single JSON response.

    SSE event types:
        ``token``      — ``{"type":"token","content":str}`` text delta.
        ``tool_call``  — ``{"type":"tool_call","tool":str,"args":dict}``.
        ``tool_result``— ``{"type":"tool_result","tool":str,"summary":str}``.
        ``done``       — Final event with full response payload.
        ``error``      — ``{"type":"error","message":str}``.

    Usage (JavaScript):
        .. code:: js

            const response = await fetch("/v1/chat/stream", { method: "POST", body: {...} });
            const reader = response.body.getReader();
            // Parse SSE frames from the byte stream.
    """
    def _generate():
        _stream_start = time.time()
        try:
            for event in processor.stream(
                query=request.query,
                top_k=request.top_k,
                temperature=request.temperature,
                include_sources=request.include_sources,
            ):
                event_type = event.get("type", "message")
                # The agent's stream dict includes "type"; SSE also carries
                # the type as the event name. Keeping it in both places is
                # harmless and makes client-side parsing simpler.
                yield f"event: {event_type}\ndata: {json.dumps(event, cls=_Encoder)}\n\n"

            get_request_metrics().record_request((time.time() - _stream_start) * 1000)

        except AllProvidersExhaustedError as e:
            logger.error(f"Chat stream exhausted: {e}")
            yield (
                "event: error\n"
                f"data: {json.dumps({'type': 'error', 'message': str(e)}, cls=_Encoder)}\n\n"
            )

        except Exception as e:
            logger.error(f"Chat stream error: {str(e)}", exc_info=True)
            get_request_metrics().record_error()
            yield (
                "event: error\n"
                f"data: {json.dumps({'type': 'error', 'message': str(e)}, cls=_Encoder)}\n\n"
            )

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
