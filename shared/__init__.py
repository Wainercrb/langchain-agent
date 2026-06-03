"""Utilities — general-purpose helpers (no RAG dependency)."""

from .correlation import get_correlation_id, set_correlation_id

__all__ = [
    "get_correlation_id",
    "set_correlation_id",
]
