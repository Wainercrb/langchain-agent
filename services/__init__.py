"""Services layer — pluggable backends (Strategy Pattern).

Lo que se swappea vive acá:
  - llm / embeddings / vector_store / parsers / logging

Razon pura (no swappable) vive en rag/ y api/.
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
