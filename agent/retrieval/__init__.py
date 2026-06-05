"""RAG retrieval services: vector search and context formatting."""

from .formatting import format_documents_as_context
from .retriever import Retriever

__all__ = ["Retriever", "format_documents_as_context"]
