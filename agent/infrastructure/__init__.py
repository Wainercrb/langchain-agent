"""Services layer — pluggable backends (Strategy Pattern).

Swappable backends live here:
  - llm / embeddings / vector_store / parsers / logging

Pure logic (not swappable) lives in rag/ and api/.
"""

from .llm import (
    GoogleProvider,
    LLMProvider,
    LLMProviderError,
)
from .embeddings import GoogleEmbeddingsWrapper
from .vector_store import VectorStore
from .parsers import FileParser, ParserFactory

__all__ = [
    # LLM Providers
    "LLMProvider",
    "LLMProviderError",
    "GoogleProvider",
    # Embeddings
    "GoogleEmbeddingsWrapper",
    # Vector Store
    "VectorStore",
    # Parsers (Strategy)
    "FileParser",
    "ParserFactory",
]
