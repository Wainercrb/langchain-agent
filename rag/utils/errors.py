"""Error handling utilities for the RAG module."""

import logging
from typing import Dict, Optional

from utils.exceptions import DocumentStoreError

logger = logging.getLogger(__name__)



def raise_document_store_error(message: str, error_code: str, details: Optional[Dict] = None):
    """Raise a standardized DocumentStoreError with context.

    Args:
        message: Error description
        error_code: Error identifier (e.g., 'DOCUMENT_INSERT_ERROR')
        details: Optional dict with error context

    Raises:
        DocumentStoreError: Always raises with provided context
    """
    logger.error(f"[{error_code}] {message}")
    raise DocumentStoreError(message=message, error_code=error_code, details=details or {})
