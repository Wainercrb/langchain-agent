import time
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from rag.core.chain import RAGChain

from .dependencies import check_health, get_feedback_service, get_rag_chain
from models import ChatRequest, ChatResponse, ErrorResponse, FeedbackRequest, HealthResponse
from services.container import logger


router = APIRouter(prefix="/v1", tags=["chat"])


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
    chain: RAGChain = Depends(get_rag_chain),
) -> ChatResponse:
    """
    Process a user query using RAG (Retrieval-Augmented Generation).

    This endpoint retrieves relevant documents based on the query, uses them as context,
    and generates a comprehensive answer via the LLM.

    Args:
        request: ChatRequest containing:
            - query: User's natural language question
            - top_k: Number of documents to retrieve (1-20)
            - include_sources: Whether to return source documents
            - temperature: LLM response creativity (0.0-1.0)
        chain: Injected RAGChain dependency for processing

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
    start_time = time.time()
    try:
        logger.info(f"Chat request: query={request.query[:50]}...")

        response = chain.invoke(
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

        return ChatResponse(
            response=response.response,
            query=request.query,
            sources=response.sources if request.include_sources else None,
            execution_time_ms=execution_time_ms,
            model=response.model,
            run_id=response.run_id,
        )

    except ValueError as e:
        logger.error(f"Chat validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_request",
                "message": "Invalid request parameters",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_error",
                "message": "Failed to process query",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


@router.post(
    "/feedback",
    status_code=200,
    responses={
        202: {"description": "Feedback accepted but LangSmith unreachable"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def feedback(
    request: FeedbackRequest,
    service=Depends(get_feedback_service),
) -> dict:
    """Record user feedback (like/dislike) correlated to a LangSmith run_id.

    Accepts feedback via the LangSmith Native Feedback API. If LangSmith is
    unreachable, the feedback is logged server-side and a 202 Accepted is
    returned instead of failing the request.

    Args:
        request: FeedbackRequest containing:
            - run_id: LangSmith run ID for feedback correlation
            - feedback_type: "like" (score=1.0) or "dislike" (score=0.0)
            - comment: Optional user comment (max 1000 chars)

    Returns:
        Dictionary with status "recorded" (200) or "accepted" (202)

    Raises:
        HTTPException (422): Validation error in request parameters
    """
    try:
        result = service.record_feedback(
            run_id=request.run_id,
            feedback_type=request.feedback_type,
            comment=request.comment,
        )
        if result.status == "accepted":
            return JSONResponse(
                content={"status": "accepted"},
                status_code=status.HTTP_202_ACCEPTED,
            )
        return {"status": "recorded"}
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_error",
                "message": "Failed to process feedback",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=200,
    responses={
        500: {"model": ErrorResponse, "description": "Health check failed"},
    },
)
async def health() -> HealthResponse:
    """
    Check service and database connectivity status.

    This endpoint performs a lightweight health check including:
    - Service availability
    - Database connectivity (Supabase/pgvector)
    - Overall system status

    Returns:
        HealthResponse containing:
            - status: "ok" if all checks pass, "error" if degraded
            - timestamp: Check timestamp
            - version: API version
            - db_connected: Database connectivity status

    Raises:
        Returns degraded status rather than raising, to allow clients to assess
        partial system health (e.g., API available but database down).
    """
    logger.debug("Health check requested")
    health_info = await check_health()

    return HealthResponse(
        status=health_info["status"],
        timestamp=datetime.utcnow(),
        db_connected=health_info.get("db_connected", False),
        llm_connected=health_info.get("llm_connected", False),
        embedding_connected=health_info.get("embedding_connected", False),
    )
