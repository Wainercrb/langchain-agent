"""Custom exception hierarchy for RAG system."""

from typing import Optional, Dict, Any


class RAGException(Exception):
    """Base exception for all RAG-related errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize RAG exception.

        Args:
            message: Error message
            error_code: Unique error code for classification
            details: Additional error details (dict)
        """
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""

    def __init__(self, message: str, error_code: str = "EMBEDDING_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class DocumentStoreError(RAGException):
    """Raised when document store operations fail."""

    def __init__(self, message: str, error_code: str = "DOCUMENT_STORE_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class IngestionError(RAGException):
    """Raised when document ingestion fails."""

    def __init__(self, message: str, error_code: str = "INGESTION_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class RetrieverError(RAGException):
    """Raised when retrieval fails."""

    def __init__(self, message: str, error_code: str = "RETRIEVER_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class FileManagerError(RAGException):
    """Raised when file management operations fail."""

    def __init__(self, message: str, error_code: str = "FILE_MANAGER_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class SchedulerError(RAGException):
    """Raised when scheduler job fails."""

    def __init__(self, message: str, error_code: str = "SCHEDULER_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class ConfigError(RAGException):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, error_code: str = "CONFIG_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)
