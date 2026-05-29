"""Vector store providers — Strategy Pattern for pluggable vector databases."""

from .supabase import VectorStore

__all__ = ["VectorStore"]
