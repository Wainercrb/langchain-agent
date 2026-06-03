"""Document and source models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class SourceDocument(BaseModel):
    """
    Metadata about a single document chunk used to generate response.

    Represents a retrieved document segment with relevance scoring and version info.

    Attributes:
        document_id: UUID of parent document
        filename: Original filename (e.g., "enrollment_guide.pdf")
        similarity_score: Cosine similarity score (0.0-1.0, higher = more relevant)
        version_date: Document version timestamp (optional, may be None if not available)
        content_preview: First ~200 chars of chunk for context
        chunk_id: UUID of specific chunk
    """

    document_id: str = Field(
        ...,
        description="UUID of parent document",
    )
    filename: str = Field(
        ...,
        description="Original filename (e.g., enrollment_guide.pdf)",
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score (0.0-1.0)",
    )
    version_date: Optional[datetime] = Field(
        default=None,
        description="Document version timestamp (optional)",
    )
    content_preview: str = Field(
        ...,
        max_length=200,
        description="First ~200 chars of chunk for context",
    )
    chunk_id: str = Field(
        ...,
        description="UUID of specific chunk",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "enrollment_guide.pdf",
                "similarity_score": 0.92,
                "version_date": "2025-01-15T10:30:00",
                "content_preview": "Enrollment Guidelines: To enroll in our program...",
                "chunk_id": "550e8400-e29b-41d4-a716-446655440001",
            }
        }
    )
