"""Abstract base for embedding providers — define el contrato para sistemas de embeddings."""

from abc import ABC, abstractmethod
from typing import List


class Embeddings(ABC):
    """Abstract base class for embedding providers.

    Cualquier embedding provider (Google, OpenAI, HuggingFace) debe implementar
    embed_query() y embed_documents().
    """

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        pass

    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        pass
