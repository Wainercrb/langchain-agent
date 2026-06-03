"""Services layer — pluggable backends (Strategy Pattern).

Swappable backends live here:
  - llm / embeddings / vector_stores / parsers / logging

Pure logic (not swappable) lives in domain/ and api/.
"""

from .llm import (
    GoogleProvider,
    LLMProvider,
    LLMProviderError,
)
from .embeddings import GoogleEmbeddingsWrapper
from .vector_stores import VectorStore, VectorStoreOps, IngestionLogger, HealthCheckable
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
    "VectorStoreOps",
    "IngestionLogger",
    "HealthCheckable",
    # Parsers (Strategy)
    "FileParser",
    "ParserFactory",
]
