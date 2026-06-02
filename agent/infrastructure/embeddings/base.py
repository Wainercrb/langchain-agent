"""Abstract base for embedding providers — defines the contract for embedding systems."""

from abc import ABC, abstractmethod
from typing import List


class Embeddings(ABC):
    """Abstract base class for embedding providers.

    Any embedding provider (Google, OpenAI, HuggingFace) must implement
    embed_query() and embed_documents().
    """

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        pass

    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        pass
