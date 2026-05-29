"""Abstract base for vector stores — define el contrato para bases de datos vectoriales."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional


class VectorStoreBase(ABC):
    """Abstract base class for vector stores.

    Cualquier vector store (Supabase pgvector, Pinecone, Qdrant) debe implementar
    estos métodos para integrarse con el sistema RAG.
    """

    @abstractmethod
    def insert_document(self, filename: str, **kwargs) -> str:
        """Insert document metadata. Returns document ID."""
        pass

    @abstractmethod
    def insert_chunks(self, document_id: str, chunks: List[Dict[str, Any]]) -> int:
        """Insert document chunks with embeddings. Returns count."""
        pass

    @abstractmethod
    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        version_filter: Optional[datetime] = None,
        latest_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.

        Args:
            query_embedding: Embedding vector of the query.
            top_k: Number of top results to return.
            version_filter: Optional minimum version date. Only documents
                with version_date >= version_filter are considered.
            latest_only: If True, only return chunks from the latest version
                of each document (determined dynamically via SQL CTE).

        Returns:
            List of result dictionaries with keys: id, document_id, text,
            chunk_index, metadata, filename, version_date, similarity_score.
        """
        pass

    @abstractmethod
    def find_document_by_hash(
        self, content_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Look up a document by its content_hash.

        Args:
            content_hash: SHA-256 hex digest of the file bytes.

        Returns:
            Document dict or None if no match found.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if vector store is healthy and connected."""
        pass

    @abstractmethod
    def log_ingestion(
        self,
        filename: str,
        status: str,
        chunk_count: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        """Log an ingestion event."""
        pass
