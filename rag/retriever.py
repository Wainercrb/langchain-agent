import logging
from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class RetrievedDocument(BaseModel):
    document_id: str = Field(...)
    chunk_id: str = Field(...)
    text: str = Field(...)
    similarity_score: float = Field(
        ..., ge=0.0, le=1.0
    )
    filename: str = Field(...)
    version_date: Optional[datetime] = Field(
        default=None
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "chunk_id": "550e8400-e29b-41d4-a716-446655440001",
                "text": "How to enroll: Step 1 is to complete the online form...",
                "similarity_score": 0.95,
                "filename": "enrollment_guide.pdf",
                "version_date": "2025-01-15T10:30:00",
            }
        }
    )


class Retriever:
    def __init__(self, vector_store, embeddings):
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
        try:
            logger.debug(
                f"Retrieve called: query={query[:50]}..., top_k={top_k}, "
                f"threshold={similarity_threshold}, version_filter={version_filter}"
            )

            query_embedding = self.embeddings.embed_query(query)
            logger.debug(f"Query embedded: {len(query_embedding)} dimensions")

            search_results = self.vector_store.search_similar(
                query_embedding=query_embedding, top_k=top_k
            )
            logger.debug(f"Vector search returned {len(search_results)} results")

            filtered_results = []
            for result in search_results:
                if result.get("similarity_score", 0) < similarity_threshold:
                    logger.debug(
                        f"Filtered out document {result.get('document_id')} "
                        f"(similarity {result.get('similarity_score')} < {similarity_threshold})"
                    )
                    continue

                if version_filter:
                    doc_version = result.get("version_date")
                    if doc_version is None:
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
