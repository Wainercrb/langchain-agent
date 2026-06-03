"""Reusable error response helpers for API endpoints."""

from datetime import datetime, timezone
from typing import Any, Optional


def validation_error_response(message: str, details: Optional[Any] = None) -> dict:
    """Build a 400 validation error response body."""
    body: dict[str, Any] = {
        "error": "invalid_request",
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if details is not None:
        body["details"] = details
    return body


def internal_error_response(message: str, error: Optional[str] = None) -> dict:
    """Build a 500 internal error response body."""
    body: dict[str, Any] = {
        "error": "internal_error",
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if error is not None:
        body["error_detail"] = error
    return body


def not_found_response(resource: str, identifier: Optional[str] = None) -> dict:
    """Build a 404 not found response body."""
    message = f"{resource} not found"
    if identifier is not None:
        message = f"No {resource.lower()} found for {identifier}"
    return {
        "error": f"{resource.lower()}_not_found",
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
