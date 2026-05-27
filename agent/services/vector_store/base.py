"""Abstract base for vector stores — define el contrato para bases de datos vectoriales."""

from abc import ABC, abstractmethod
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
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
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
