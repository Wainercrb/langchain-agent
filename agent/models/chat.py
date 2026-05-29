"""Chat request and input models."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    """
    Request model for POST /v1/chat endpoint.

    Validates user query and retrieval parameters for RAG chain processing.

    Attributes:
        query: User query to search documents (1-2000 chars, non-empty)
        top_k: Number of documents to retrieve (1-20, default 5)
        include_sources: Whether to include source documents in response (default True)
        temperature: LLM temperature for response creativity (0.0-1.0, default 0.7)
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User query to search documents",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve (1-20)",
    )
    include_sources: bool = Field(
        default=True,
        description="Include source documents in response",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="LLM temperature (0.0=deterministic, 1.0=creative)",
    )
    latest_only: bool = Field(
        default=True,
        description="Only retrieve latest document versions",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What are the enrollment requirements?",
                "top_k": 5,
                "include_sources": True,
                "temperature": 0.7,
                "latest_only": True,
            }
        }
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v):
        """Ensure query is not empty or whitespace-only."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty or whitespace-only")
        return v.strip()
