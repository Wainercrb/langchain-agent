"""Exception hierarchy and error helpers for the RAG system."""

from enum import Enum
from typing import Any, Dict, Optional


class Severity(str, Enum):
    """Alert severity levels, ordered from least to most critical."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


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

    @property
    def severity(self) -> Severity:
        """Derive alert severity from exception type.

        Transient errors (retriable) → WARNING.
        Permanent errors (auth, config) → ERROR.
        Base RAGException defaults to ERROR.
        """
        return Severity.ERROR

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


class LLMProviderError(RAGException):
    """Base exception for LLM provider errors, unified under RAGException."""

    def __init__(
        self,
        message: str,
        provider: str,
        original_error: Optional[Exception] = None,
        is_transient: bool = True,
        error_code: str = "LLM_PROVIDER_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.provider = provider
        self.original_error = original_error
        self.is_transient = is_transient
        super().__init__(message, error_code, details or {"provider": provider})


class TransientLLMError(LLMProviderError):
    """LLM error that SHOULD be retried (rate limits, timeouts, 5xx)."""

    def __init__(
        self, message: str, provider: str, original_error: Optional[Exception] = None
    ):
        super().__init__(
            message,
            provider,
            original_error,
            is_transient=True,
            error_code="LLM_TRANSIENT_ERROR",
        )

    @property
    def severity(self) -> Severity:
        return Severity.WARNING


class PermanentLLMError(LLMProviderError):
    """LLM error that should NOT be retried (auth failures, invalid model, 4xx)."""

    def __init__(
        self, message: str, provider: str, original_error: Optional[Exception] = None
    ):
        super().__init__(
            message,
            provider,
            original_error,
            is_transient=False,
            error_code="LLM_PERMANENT_ERROR",
        )

    @property
    def severity(self) -> Severity:
        return Severity.ERROR


class AllProvidersExhaustedError(LLMProviderError):
    """Raised when all LLM providers in the failover chain are exhausted."""

    def __init__(
        self,
        message: str,
        attempted_providers: list[str],
        errors: list[Exception],
    ):
        self.attempted_providers = attempted_providers
        self.errors = errors
        super().__init__(
            message,
            provider="all",
            error_code="ALL_PROVIDERS_EXHAUSTED",
            is_transient=False,
        )

    @property
    def severity(self) -> Severity:
        return Severity.CRITICAL
