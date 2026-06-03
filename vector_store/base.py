"""Abstract base for vector stores — defines the contract for vector databases.

Split into focused ABCs per ISP (Interface Segregation Principle):
- VectorStoreOps: core vector operations
- IngestionLogger: ingestion logging concern
- HealthCheckable: operational health check concern
- VectorStore: composite convenience interface combining all three
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional


class VectorStoreOps(ABC):
    """Core vector operations (search, insert, lookup)."""

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
    def find_document_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Look up a document by its content_hash.

        Args:
            content_hash: SHA-256 hex digest of the file bytes.

        Returns:
            Document dict or None if no match found.
        """
        pass


class IngestionLogger(ABC):
    """Ingestion logging concern."""

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


class HealthCheckable(ABC):
    """Operational health check concern."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if vector store is healthy and connected."""
        pass


class VectorStore(VectorStoreOps, IngestionLogger, HealthCheckable, ABC):
    """Full vector store interface (combines all 3 ABCs).

    Backward-compatible: implements the union of VectorStoreOps,
    IngestionLogger, and HealthCheckable. Concrete implementations
    like Supabase's VectorStore inherit from this composite.
    """
    pass


# Backward-compatible alias for code still referencing the old name
VectorStoreBase = VectorStore
