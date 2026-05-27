import logging
import time
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from rag.rag_chain import RAGChain

from .dependencies import check_health, get_rag_chain
from .models import ChatRequest, ChatResponse, HealthResponse, ErrorResponse

logger = logging.getLogger(__name__)

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
        )

    except ValueError as e:
        logger.error(f"Chat validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_request",
                "message": str(e),
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
    )
