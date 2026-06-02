"""Retrieval and internal document models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class RetrievedDocument(BaseModel):
    """Internal model for retrieved document chunks."""

    document_id: str = Field(...)
    chunk_id: str = Field(...)
    text: str = Field(...)
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    filename: str = Field(...)
    version_date: Optional[datetime] = Field(default=None)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "chunk_id": "550e8400-e29b-41d4-a716-446655440001",
                "text": "How to enroll: Step 1 is to complete the online form...",
                "similarity_score": 0.95,
                "filename": "enrollment_guide.pdf",
                "version_date": "2025-01-15T10:30:00",
            }
        }
    )
