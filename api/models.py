"""Request and response models for the Retrieval API.

All models use Pydantic for validation, serialization, and OpenAPI schema generation.
"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Request model for POST /v1/chat endpoint.

    Validates user query, retrieval parameters, and LLM settings.

    Validation Rules:
    - query: Required, non-empty, max 2000 chars (prevent token abuse)
    - top_k: 1-20 documents (balance precision + latency)
    - temperature: 0.0-1.0 (LLM creativity level)
    - include_sources: Include retrieved documents in response
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User query to search documents (1-2000 characters)",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve (1-20, default 5)",
    )
    include_sources: bool = Field(
        default=True, description="Include source documents in response (default True)"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="LLM temperature (0.0=deterministic, 1.0=creative, default 0.7)",
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v):
        """Ensure query is not empty or whitespace-only.

        Args:
            v: Query string.

        Returns:
            Stripped query string.

        Raises:
            ValueError: If query is empty or whitespace-only.
        """
        if not v or not v.strip():
            raise ValueError("Query cannot be empty or whitespace-only")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the enrollment requirements?",
                "top_k": 5,
                "include_sources": True,
                "temperature": 0.7,
            }
        }


class SourceDocument(BaseModel):
    """Metadata about a document chunk used to generate response.

    Fields:
    - document_id: UUID of parent document
    - filename: Original filename (e.g., "enrollment_guide.pdf")
    - similarity_score: Cosine similarity (0.0-1.0), higher = more relevant
    - version_date: Document version timestamp
    - content_preview: First ~200 chars of chunk (for context)
    - chunk_id: UUID of specific chunk
    """

    document_id: str = Field(..., description="UUID of parent document")
    filename: str = Field(..., description="Original filename (e.g., enrollment_guide.pdf)")
    similarity_score: float = Field(
        ge=0.0, le=1.0, description="Cosine similarity score (0.0-1.0)"
    )
    version_date: Optional[datetime] = Field(default=None, description="Document version date (ISO format, optional)")
    content_preview: str = Field(
        max_length=200, description="First 200 chars of chunk for context"
    )
    chunk_id: str = Field(..., description="UUID of specific chunk")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "enrollment_guide.pdf",
                "similarity_score": 0.92,
                "version_date": "2025-01-15T10:30:00",
                "content_preview": "Enrollment Guidelines: To enroll in our program...",
                "chunk_id": "550e8400-e29b-41d4-a716-446655440001",
            }
        }


class ChatResponse(BaseModel):
    """Response model for POST /v1/chat endpoint.

    Fields:
    - response: LLM-generated answer
    - query: Echo of user's original query
    - sources: Retrieved documents (if include_sources=True)
    - execution_time_ms: Total roundtrip time in milliseconds
    - model: LLM model used (e.g., "gemini-2.5-flash")
    """

    response: str = Field(..., description="LLM-generated answer to user query")
    query: str = Field(..., description="Echo of user's original query")
    sources: Optional[List[SourceDocument]] = Field(
        default=None, description="Retrieved documents used for context (if requested)"
    )
    execution_time_ms: float = Field(..., description="Total query time in milliseconds")
    model: str = Field(
        default="gemini-2.5-flash", description="LLM model used to generate response"
    )

    class Config:
        json_schema_extra = {
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
            }
        }


class HealthResponse(BaseModel):
    """Response model for GET /v1/health endpoint.

    Indicates overall service health and connectivity status.
    """

    status: str = Field(..., description="Service status: 'ok' or 'error'")
    timestamp: datetime = Field(..., description="Server timestamp (ISO format)")
    version: str = Field(default="1.0.0", description="API version")
    db_connected: bool = Field(..., description="Whether database connection is healthy")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "timestamp": "2025-01-15T12:00:00",
                "version": "1.0.0",
                "db_connected": True,
            }
        }


class ErrorResponse(BaseModel):
    """Response model for error responses (400, 500, etc).

    Provides structured error information for API consumers.
    """

    error: str = Field(
        ...,
        description="Machine-readable error code (e.g., 'query_too_long', 'internal_error')",
    )
    message: str = Field(..., description="Human-readable error message")
    timestamp: datetime = Field(..., description="Error timestamp (ISO format)")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "query_too_long",
                "message": "Query exceeds maximum length of 2000 characters",
                "timestamp": "2025-01-15T12:00:00",
            }
        }
