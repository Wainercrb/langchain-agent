"""Supabase pgvector vector store — implementa VectorStoreBase usando Supabase."""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from services.vector_store.base import VectorStoreBase
from utils.exceptions import DocumentStoreError
from services.logging import logger


class VectorStore(VectorStoreBase):
    """Manage vector operations in Supabase pgvector."""

    def __init__(self, supabase_client):
        self.client = supabase_client
        logger.info("VectorStore initialized")

    def insert_document(
        self,
        filename: str,
        version_date: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        try:
            version_date = (
                version_date.isoformat()
                if isinstance(version_date, datetime)
                else (version_date or datetime.utcnow().isoformat())
            )
            doc_id = str(uuid.uuid4())
            metadata = metadata or {}

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
            raise DocumentStoreError(
                message=f"Failed to insert document {filename}: {str(e)}",
                error_code="DOCUMENT_INSERT_ERROR",
                details={"filename": filename},
            )

    def insert_chunks(self, document_id: str, chunks: List[Dict[str, Any]]) -> int:
        try:
            chunk_records = [
                {
                    "id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "chunk_index": idx,
                    "text": chunk["text"],
                    "embedding": chunk["embedding"],
                    "metadata": chunk.get("metadata", {}),
                    "created_at": datetime.utcnow().isoformat(),
                }
                for idx, chunk in enumerate(chunks)
            ]

            self.client.table("document_chunks").insert(chunk_records).execute()
            logger.info(f"Inserted {len(chunk_records)} chunks for document {document_id}")
            return len(chunk_records)
        except Exception as e:
            raise DocumentStoreError(
                message=f"Failed to insert chunks: {str(e)}",
                error_code="CHUNKS_INSERT_ERROR",
                details={"document_id": document_id, "chunk_count": len(chunks)},
            )

    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        version_filter: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        try:
            rpc_params = {
                "query_embedding": query_embedding,
                "top_k": top_k,
            }
            if version_filter is not None:
                rpc_params["version_filter"] = (
                    version_filter.isoformat()
                    if isinstance(version_filter, datetime)
                    else version_filter
                )

            response = self.client.rpc("search_similar_chunks", rpc_params).execute()

            results = response.data or []

            top_score = f"{results[0]['similarity_score']:.3f}" if results else "N/A"
            logger.info(
                f"Found {len(results)} similar chunks "
                f"(top_k={top_k}, top_score={top_score}, version_filter={version_filter})"
            )
            return results
        except Exception as e:
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
        try:
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
            raise DocumentStoreError(
                message=f"Failed to log ingestion: {str(e)}",
                error_code="INGESTION_LOG_ERROR",
            )

    async def health_check(self) -> bool:
        try:
            response = self.client.table("documents").select("id", count="exact").limit(1).execute()
            logger.info("Health check passed: database is healthy")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
