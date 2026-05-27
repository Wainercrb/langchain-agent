"""Abstract base classes for RAG components - defines contracts for extension."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Embeddings(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string.

        Args:
            query: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        pass

    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed multiple documents.

        Args:
            documents: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass


class VectorStoreBase(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def insert_document(self, filename: str, **kwargs) -> str:
        """Insert document metadata.

        Returns:
            Document ID
        """
        pass

    @abstractmethod
    def insert_chunks(self, document_id: str, chunks: List[Dict[str, Any]]) -> int:
        """Insert document chunks with embeddings.

        Returns:
            Number of chunks inserted
        """
        pass

    @abstractmethod
    def search_similar(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return

        Returns:
            List of similar documents with scores
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if vector store is healthy and connected.

        Returns:
            True if connected and operational, False otherwise
        """
        pass


class FileParser(ABC):
    """Abstract base class for file parsers."""

    @abstractmethod
    def can_parse(self, file_extension: str) -> bool:
        """Check if this parser can handle the file type.

        Args:
            file_extension: File extension (e.g., '.pdf')

        Returns:
            True if parser can handle this extension
        """
        pass

    @abstractmethod
    def parse(self, file_path: str) -> str:
        """Parse file and extract text.

        Args:
            file_path: Path to file

        Returns:
            Extracted text content

        Raises:
            Exception: If parsing fails
        """
        pass
