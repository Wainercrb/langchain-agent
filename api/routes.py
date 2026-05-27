"""FastAPI route handlers for the Retrieval API.

Implements two endpoints:
- POST /v1/chat: Query documents with RAG
- GET /v1/health: Service health check
"""

import time
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime

from .models import ChatRequest, ChatResponse, HealthResponse, ErrorResponse
from .dependencies import get_rag_chain, check_health
from rag.rag_chain import RAGChain

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse, status_code=200)
async def chat(request: ChatRequest, chain: RAGChain = Depends(get_rag_chain)):
    """Query documents with RAG context.

    Accepts a user query, retrieves relevant documents, and generates an
    LLM-powered answer augmented with document context. Returns the answer
    along with source documents and execution timing.

    Args:
        request: ChatRequest with query, top_k, temperature, include_sources.
        chain: RAGChain instance (injected by FastAPI).

    Returns:
        ChatResponse with answer, sources (if requested), and timing.

    Raises:
        HTTPException: 400 for validation errors, 500 for server errors.

    Example:
        >>> response = await chat(
        ...     ChatRequest(query="How to enroll?", top_k=5),
        ...     chain=rag_chain
        ... )
        >>> print(response.response)
        'To enroll, you need to...'
    """
    start_time = time.time()
    try:
        logger.info(f"Chat request: query={request.query[:50]}...")

        # Invoke RAG chain
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


@router.get("/health", response_model=HealthResponse, status_code=200)
async def health():
    """Health check endpoint.

    Verifies service connectivity and returns overall health status.
    Used by load balancers and monitoring systems.

    Returns:
        HealthResponse with status ("ok" or "error") and db_connected flag.

    Example:
        >>> response = await health()
        >>> print(response.status)
        'ok'
        >>> print(response.db_connected)
        True
    """
    logger.debug("Health check requested")
    health_info = await check_health()

    return HealthResponse(
        status=health_info["status"],
        timestamp=datetime.utcnow(),
        db_connected=health_info.get("db_connected", False),
    )


@router.post("/debug/retrieve", response_model=dict, status_code=200)
async def debug_retrieve(request: ChatRequest):
    """Debug endpoint: Show retrieved documents without LLM processing.

    Useful for troubleshooting retrieval quality and similarity scores.

    Args:
        request: ChatRequest with query and top_k.

    Returns:
        Dictionary with query, retrieved documents, and similarity scores.
    """
    try:
        from api.dependencies import get_retriever
        
        logger.info(f"Debug retrieve: query={request.query}, top_k={request.top_k}")
        
        retriever = get_retriever()
        retrieved = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=0.0,  # No filtering for debug
        )
        
        results = [
            {
                "document_id": doc.document_id,
                "filename": doc.filename,
                "similarity_score": doc.similarity_score,
                "text_preview": doc.text[:200] + "..." if len(doc.text) > 200 else doc.text,
            }
            for doc in retrieved
        ]
        
        return {
            "query": request.query,
            "top_k": request.top_k,
            "results_count": len(results),
            "results": results,
        }
    except Exception as e:
        logger.error(f"Debug retrieve error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "debug_error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
