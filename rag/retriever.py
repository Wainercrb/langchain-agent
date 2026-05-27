"""Document retrieval from vector store based on semantic similarity."""

from typing import List, Optional
from datetime import datetime, date
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RetrievedDocument(BaseModel):
    """A document chunk retrieved from the vector store."""

    document_id: str = Field(..., description="UUID of parent document")
    chunk_id: str = Field(..., description="UUID of specific chunk")
    text: str = Field(..., description="Chunk text content")
    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Cosine similarity score (0.0-1.0)"
    )
    filename: str = Field(..., description="Original filename")
    version_date: Optional[datetime] = Field(default=None, description="Document version date (optional)")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "chunk_id": "550e8400-e29b-41d4-a716-446655440001",
                "text": "How to enroll: Step 1 is to complete the online form...",
                "similarity_score": 0.95,
                "filename": "enrollment_guide.pdf",
                "version_date": "2025-01-15T10:30:00",
            }
        }


class Retriever:
    """Retrieve documents from vector store based on query similarity.

    The Retriever accepts a query string, embeds it using the same embedding
    model as the documents, and searches the pgvector store for similar chunks.
    Results are filtered by similarity threshold and optionally by document
    version date.

    Attributes:
        vector_store: VectorStore instance for pgvector operations.
        embeddings: GoogleEmbeddingsWrapper for query embedding.
    """

    def __init__(self, vector_store, embeddings):
        """Initialize Retriever.

        Args:
            vector_store: VectorStore instance for similarity search.
            embeddings: GoogleEmbeddingsWrapper for embedding queries.
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        logger.info("Retriever initialized")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        version_filter: Optional[date] = None,
    ) -> List[RetrievedDocument]:
        """Retrieve relevant documents for a user query.

        Embeds the query, searches the vector store, applies threshold and
        version filters, and returns results ordered by similarity (descending).

        Args:
            query: User query string to search for.
            top_k: Maximum number of top results to return (1-20, default 5).
            similarity_threshold: Minimum similarity score to include (0.0-1.0,
                default 0.5). Results below this threshold are filtered out.
            version_filter: Optional document version date (ISO date). Only
                documents with version_date >= version_filter are included.

        Returns:
            List of RetrievedDocument objects ordered by similarity_score
            (highest first). Empty list if no results found.

        Raises:
            Exception: If vector store search fails or embedding fails.
                Details are logged with full traceback.
        """
        try:
            logger.debug(
                f"Retrieve called: query={query[:50]}..., top_k={top_k}, "
                f"threshold={similarity_threshold}, version_filter={version_filter}"
            )

            # 1. Embed the query
            query_embedding = self.embeddings.embed_query(query)
            logger.debug(f"Query embedded: {len(query_embedding)} dimensions")

            # 2. Search pgvector
            search_results = self.vector_store.search_similar(
                query_embedding=query_embedding, top_k=top_k
            )
            logger.debug(f"Vector search returned {len(search_results)} results")

            # 3. Filter by threshold and version
            filtered_results = []
            for result in search_results:
                # Check similarity threshold
                if result.get("similarity_score", 0) < similarity_threshold:
                    logger.debug(
                        f"Filtered out document {result.get('document_id')} "
                        f"(similarity {result.get('similarity_score')} < {similarity_threshold})"
                    )
                    continue

                # Check version filter
                if version_filter:
                    doc_version = result.get("version_date")
                    if doc_version is None:
                        # Skip documents with no version date when version filtering
                        logger.debug(
                            f"Filtered out document {result.get('document_id')} "
                            f"(version_date is None)"
                        )
                        continue
                    if isinstance(doc_version, datetime):
                        doc_version = doc_version.date()
                    if doc_version < version_filter:
                        logger.debug(
                            f"Filtered out document {result.get('document_id')} "
                            f"(version {doc_version} < {version_filter})"
                        )
                        continue

                filtered_results.append(result)

            # 4. Format and return
            retrieved_documents = []
            for result in filtered_results:
                doc = RetrievedDocument(
                    document_id=result.get("document_id", ""),
                    chunk_id=result.get("id", ""),
                    text=result.get("text", ""),
                    similarity_score=result.get("similarity_score", 0.0),
                    filename=result.get("filename", "unknown"),
                    version_date=result.get("version_date"),
                )
                retrieved_documents.append(doc)

            logger.info(
                f"Retrieve complete: query={query[:50]}..., returned {len(retrieved_documents)} documents"
            )
            return retrieved_documents

        except Exception as e:
            logger.error(
                f"Retriever.retrieve failed: query={query[:50]}..., error={str(e)}",
                exc_info=True,
            )
            raise
