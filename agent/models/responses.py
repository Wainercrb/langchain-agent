"""Response models for API and RAG operations."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .document import SourceDocument


class MetricsResponse(BaseModel):
    """
    Response model for GET /v1/metrics endpoint.

    Provides lightweight operational counters. LangSmith handles the
    comprehensive observability dashboards.

    Attributes:
        request_count: Total number of requests handled since startup
        error_count: Total number of errors since startup
        avg_latency_ms: Average request latency in milliseconds
        langsmith_dashboard_url: Link to LangSmith project dashboard (if configured)
    """

    request_count: int = Field(
        default=0,
        ge=0,
        description="Total number of requests handled since startup",
    )
    error_count: int = Field(
        default=0,
        ge=0,
        description="Total number of errors since startup",
    )
    avg_latency_ms: float = Field(
        default=0.0,
        ge=0,
        description="Average request latency in milliseconds",
    )
    total_input_tokens: int = Field(
        default=0,
        ge=0,
        description="Total input (prompt) tokens since startup",
    )
    total_output_tokens: int = Field(
        default=0,
        ge=0,
        description="Total output (completion) tokens since startup",
    )
    avg_tokens_per_request: float = Field(
        default=0.0,
        ge=0.0,
        description="Average total tokens per request (input + output)",
    )
    langsmith_dashboard_url: Optional[str] = Field(
        default=None,
        description="Link to LangSmith project dashboard (null if tracing disabled)",
    )
    langsmith_audit_url: Optional[str] = Field(
        default=None,
        description="Link to LangSmith audit/project URL (null if tracing disabled)",
    )


class ChatResponse(BaseModel):
    """
    Response model for POST /v1/chat endpoint.

    Contains LLM-generated answer with optional source documents and execution metrics.

    Attributes:
        response: LLM-generated answer based on retrieved context
        query: Echo of user's original query
        sources: List of retrieved documents used for context (None if include_sources=False)
        execution_time_ms: Total roundtrip execution time in milliseconds (≥0)
        model: LLM model identifier (default: "gemini-2.5-flash")
    """

    response: str = Field(
        ...,
        description="LLM-generated answer",
    )
    query: str = Field(
        ...,
        description="Echo of user's original query",
    )
    sources: Optional[List[SourceDocument]] = Field(
        default=None,
        description="Retrieved documents used for context",
    )
    execution_time_ms: float = Field(
        ...,
        ge=0,
        description="Total roundtrip execution time in milliseconds",
    )
    model: str = Field(
        default="gemini-2.5-flash",
        description="LLM model identifier",
    )
    run_id: Optional[str] = Field(
        default=None,
        description="LangSmith run ID for feedback correlation (null when tracing disabled)",
    )
    usage_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Token usage metadata from LLM provider (null when unavailable)",
    )
    llm_latency_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="Time spent in the LLM call only (excludes retrieval, serialization). Approximates TTFT for non-streaming calls.",
    )
    langsmith_tags: Optional[List[str]] = Field(
        default=None,
        description="Tags applied to the LangSmith run for filtering (null when tracing disabled)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response": "To enroll in the program, you need to follow these steps: 1. Complete the online form...",
                "query": "What are the enrollment requirements?",
                "sources": [
                    {
                        "document_id": "550e8400-e29b-41d4-a716-446655440000",
                        "filename": "enrollment_guide.pdf",
                        "similarity_score": 0.92,
                        "version_date": "2025-01-15T10:30:00",
                        "content_preview": "Enrollment Guidelines: To enroll in our program...",
                        "chunk_id": "550e8400-e29b-41d4-a716-446655440001",
                    }
                ],
                "execution_time_ms": 2340.5,
                "model": "gemini-2.5-flash",
                "run_id": "550e8400-e29b-41d4-a716-446655440000",
                "llm_latency_ms": 1850.3,
                "langsmith_tags": ["model:openai/gpt-4o", "agent:tool-calling", "top_k:5"],
            }
        }
    )


class HealthResponse(BaseModel):
    """
    Response model for GET /v1/health endpoint.

    Indicates service operational status and database connectivity.

    Attributes:
        status: Service status ("ok" or "error")
        timestamp: Health check timestamp
        version: API version (default: "1.0.0")
        db_connected: Whether database is reachable
    """

    status: str = Field(
        ...,
        pattern="^(ok|error)$",
        description="Service status: 'ok' or 'error'",
    )
    timestamp: datetime = Field(
        ...,
        description="Health check timestamp",
    )
    version: str = Field(
        default="1.0.0",
        description="API version",
    )
    db_connected: bool = Field(
        ...,
        description="Database connectivity status",
    )
    llm_connected: bool = Field(
        default=False,
        description="LLM provider availability",
    )
    embedding_connected: bool = Field(
        default=False,
        description="Embedding service availability",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "timestamp": "2025-01-15T12:00:00",
                "version": "1.0.0",
                "db_connected": True,
            }
        }
    )


class ErrorResponse(BaseModel):
    """
    Standard error response for all error statuses.

    Provides structured error information with code and human-readable message.

    Error Codes:
        - "query_too_long": Query exceeds 2000 character limit
        - "invalid_top_k": top_k outside 1-20 range
        - "invalid_temperature": temperature outside 0.0-1.0 range
        - "db_error": Database connection failure
        - "embedding_error": Embedding service failure
        - "llm_error": LLM API error
        - "internal_error": Unexpected server error

    Attributes:
        error: Error code (machine-readable identifier)
        message: Human-readable error description
        timestamp: Error occurrence timestamp
    """

    error: str = Field(
        ...,
        description="Error code (machine-readable): query_too_long, invalid_top_k, invalid_temperature, db_error, embedding_error, llm_error, internal_error",
    )
    message: str = Field(
        ...,
        description="Human-readable error message",
    )
    timestamp: datetime = Field(
        ...,
        description="Error timestamp",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "query_too_long",
                "message": "Query exceeds maximum length of 2000 characters",
                "timestamp": "2025-01-15T12:00:00",
            }
        }
    )
