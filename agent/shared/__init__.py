"""Shared — cross-cutting helpers (no RAG dependency)."""

from .correlation import get_correlation_id, set_correlation_id
from .exceptions import (
    AllProvidersExhaustedError,
    DocumentStoreError,
    EmbeddingError,
    IngestionError,
    LLMProviderError,
    PermanentLLMError,
    RAGException,
    Severity,
    TransientLLMError,
)
from .filters import filter_by_threshold

__all__ = [
    "get_correlation_id",
    "set_correlation_id",
    "AllProvidersExhaustedError",
    "DocumentStoreError",
    "EmbeddingError",
    "IngestionError",
    "LLMProviderError",
    "PermanentLLMError",
    "RAGException",
    "Severity",
    "TransientLLMError",
    "filter_by_threshold",
]
