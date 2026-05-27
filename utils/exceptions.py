"""Custom exception hierarchy for RAG system."""

from typing import Any, Dict, Optional


class RAGException(Exception):
    """Base exception for all RAG-related errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""

    def __init__(
        self,
        message: str,
        error_code: str = "EMBEDDING_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details)


class DocumentStoreError(RAGException):
    """Raised when document store operations fail."""

    def __init__(
        self,
        message: str,
        error_code: str = "DOCUMENT_STORE_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details)


class IngestionError(RAGException):
    """Raised when document ingestion fails."""

    def __init__(
        self,
        message: str,
        error_code: str = "INGESTION_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details)



