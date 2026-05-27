"""Vector store operations using Supabase pgvector."""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import VectorStoreBase

logger = logging.getLogger(__name__)


class VectorStore(VectorStoreBase):
    """Manage vector operations in Supabase pgvector."""

    def __init__(self, supabase_client):
        self.client = supabase_client
        self.offline_mode = supabase_client is None

        if self.offline_mode:
            logger.warning("VectorStore initialized in OFFLINE MODE (no database connection)")
        else:
            logger.info("VectorStore initialized")

    def insert_document(
        self,
        filename: str,
        version_date: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        from utils.exceptions import DocumentStoreError

        try:
            if version_date is None:
                version_date = datetime.utcnow().isoformat()
            else:
                version_date = (
                    version_date.isoformat() if isinstance(version_date, datetime) else version_date
                )

            doc_id = str(uuid.uuid4())
            metadata = metadata or {}

            if self.offline_mode:
                logger.info(f"[OFFLINE] Would insert document: {filename} (id={doc_id})")
                return doc_id

            self.client.table("documents").insert(
                {
                    "id": doc_id,
                    "filename": filename,
                    "version_date": version_date,
                    "metadata": metadata,
                    "created_at": datetime.utcnow().isoformat(),
                }
            ).execute()

            logger.info(f"Inserted document: {filename} (id={doc_id})")
            return doc_id
        except Exception as e:
            logger.error(f"Failed to insert document: {str(e)}")
            raise DocumentStoreError(
                message=f"Failed to insert document {filename}: {str(e)}",
                error_code="DOCUMENT_INSERT_ERROR",
                details={"filename": filename, "error": str(e)},
            )

    def insert_chunks(self, document_id: str, chunks: List[Dict[str, Any]]) -> int:
        from utils.exceptions import DocumentStoreError

        try:
            if self.offline_mode:
                logger.info(
                    f"[OFFLINE] Would insert {len(chunks)} chunks for document {document_id}"
                )
                return len(chunks)

            chunk_records = []
            for idx, chunk in enumerate(chunks):
                chunk_records.append(
                    {
                        "id": str(uuid.uuid4()),
                        "document_id": document_id,
                        "chunk_index": idx,
                        "text": chunk["text"],
                        "embedding": chunk["embedding"],
                        "metadata": chunk.get("metadata", {}),
                        "created_at": datetime.utcnow().isoformat(),
                    }
                )

            self.client.table("document_chunks").insert(chunk_records).execute()
            logger.info(f"Inserted {len(chunk_records)} chunks for document {document_id}")
            return len(chunk_records)
        except Exception as e:
            logger.error(f"Failed to insert chunks: {str(e)}")
            raise DocumentStoreError(
                message=f"Failed to insert chunks: {str(e)}",
                error_code="CHUNKS_INSERT_ERROR",
                details={"document_id": document_id, "chunk_count": len(chunks)},
            )

    def get_latest_document(self, filename: str) -> Optional[Dict[str, Any]]:

        from utils.exceptions import DocumentStoreError

        try:
            response = (
                self.client.table("documents")
                .select("*")
                .eq("filename", filename)
                .order("version_date", desc=True)
                .limit(1)
                .execute()
            )

            if response.data:
                logger.info(
                    f"Found latest document: {filename} (version_date={response.data[0].get('version_date')})"
                )
                return response.data[0]

            logger.warning(f"No document found: {filename}")
            return None
        except Exception as e:
            logger.error(f"Failed to get document: {str(e)}")
            raise DocumentStoreError(
                message=f"Failed to retrieve document {filename}: {str(e)}",
                error_code="DOCUMENT_RETRIEVE_ERROR",
            )

    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        from utils.exceptions import DocumentStoreError

        try:
            if self.offline_mode:
                logger.info(f"[OFFLINE] Would search for {top_k} similar chunks")
                return []

            # Use Supabase PostgREST API to fetch chunks with embeddings
            # We'll compute similarity client-side and sort
            response = (
                self.client.table("document_chunks")
                .select(
                    "id, document_id, text, chunk_index, metadata, embedding, documents(filename, version_date)"
                )
                .execute()
            )

            # Compute cosine similarity for each chunk
            import numpy as np

            similar_chunks = []

            for item in response.data or []:
                # Parse embedding from pgvector format
                embedding = item.get("embedding")
                if not embedding:
                    continue

                # Parse if it's a string representation
                if isinstance(embedding, str):
                    # Remove brackets and split by comma
                    embedding = np.array([float(x) for x in embedding.strip("[]").split(",")])
                else:
                    embedding = np.array(embedding)

                # Compute cosine similarity: (A·B) / (||A|| ||B||)
                dot_product = np.dot(query_embedding, embedding)
                norm_query = np.linalg.norm(query_embedding)
                norm_chunk = np.linalg.norm(embedding)

                if norm_query > 0 and norm_chunk > 0:
                    similarity = dot_product / (norm_query * norm_chunk)
                else:
                    similarity = 0.0

                # Get document info
                documents_list = item.get("documents")
                if (
                    not documents_list
                    or not isinstance(documents_list, list)
                    or len(documents_list) == 0
                ):
                    documents_list = [{}]

                doc_info = documents_list[0] if isinstance(documents_list, list) else {}

                similar_chunks.append(
                    {
                        "id": item.get("id"),
                        "document_id": item.get("document_id"),
                        "text": item.get("text"),
                        "chunk_index": item.get("chunk_index"),
                        "metadata": item.get("metadata"),
                        "filename": doc_info.get("filename", "unknown"),
                        "version_date": doc_info.get("version_date"),
                        "similarity_score": similarity,
                    }
                )

            # Sort by similarity descending and filter by threshold
            similar_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
            flattened_results = [
                chunk
                for chunk in similar_chunks
                if chunk["similarity_score"] >= similarity_threshold
            ][:top_k]

            # Format top score for logging
            top_score = (
                f"{flattened_results[0]['similarity_score']:.3f}" if flattened_results else "N/A"
            )
            logger.info(
                f"Found {len(flattened_results)} similar chunks "
                f"(top_k={top_k}, threshold={similarity_threshold}, top_score={top_score})"
            )
            return flattened_results
        except Exception as e:
            logger.error(f"Failed to search similar chunks: {str(e)}")
            raise DocumentStoreError(
                message=f"Failed to search similar chunks: {str(e)}",
                error_code="SEARCH_ERROR",
            )

    def log_ingestion(
        self,
        filename: str,
        status: str,
        chunk_count: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        from utils.exceptions import DocumentStoreError

        try:
            if self.offline_mode:
                logger.info(f"[OFFLINE] Would log: {filename} ({status}, {chunk_count} chunks)")
                return

            self.client.table("ingestion_logs").insert(
                {
                    "id": str(uuid.uuid4()),
                    "filename": filename,
                    "status": status,
                    "chunk_count": chunk_count,
                    "error_message": error_message,
                    "processed_at": datetime.utcnow().isoformat(),
                }
            ).execute()

            logger.info(f"Logged ingestion: {filename} ({status}, {chunk_count} chunks)")
        except Exception as e:
            logger.error(f"Failed to log ingestion: {str(e)}")
            raise DocumentStoreError(
                message=f"Failed to log ingestion: {str(e)}",
                error_code="INGESTION_LOG_ERROR",
            )

    async def health_check(self) -> bool:
        """
        Check vector store health and database connectivity.

        Performs a lightweight query to verify:
        - Database connection is alive
        - Supabase client is operational
        - Minimal latency acceptable

        Returns:
            bool: True if healthy, False if offline or connection failed

        Note:
            In offline_mode (no database), returns True to allow graceful degradation.
        """
        try:
            if self.offline_mode:
                logger.info("Health check: OFFLINE mode - returning True (no database)")
                return True

            # Lightweight query: just count documents, no data transfer
            response = self.client.table("documents").select("id", count="exact").limit(1).execute()
            logger.info("Health check passed: database is healthy")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
