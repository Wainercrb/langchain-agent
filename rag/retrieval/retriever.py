import logging
from datetime import datetime
from typing import List, Optional

from models import RetrievedDocument

from ..utils import filter_by_threshold, filter_by_version

logger = logging.getLogger(__name__)


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
        version_filter: Optional[datetime] = None,
    ) -> List[RetrievedDocument]:
        try:
            logger.debug(
                f"Retrieve called: query={query[:50]}..., top_k={top_k}, "
                f"threshold={similarity_threshold}, version_filter={version_filter}"
            )

            query_embedding = self.embeddings.embed_query(query)
            search_results = self.vector_store.search_similar(
                query_embedding=query_embedding, top_k=top_k
            )

            filtered = list(filter_by_threshold(search_results, similarity_threshold))
            if version_filter:
                filtered = list(filter_by_version(filtered, version_filter))

            retrieved_documents = [
                RetrievedDocument(
                    document_id=result.get("document_id", ""),
                    chunk_id=result.get("id", ""),
                    text=result.get("text", ""),
                    similarity_score=result.get("similarity_score", 0.0),
                    filename=result.get("filename", "unknown"),
                    version_date=result.get("version_date"),
                )
                for result in filtered
            ]

            logger.info(f"Retrieve complete: returned {len(retrieved_documents)} documents")
            return retrieved_documents

        except Exception as e:
            logger.error(f"Retriever.retrieve failed: {str(e)}", exc_info=True)
            raise
