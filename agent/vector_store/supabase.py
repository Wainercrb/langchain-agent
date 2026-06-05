"""Supabase pgvector vector store — implements VectorStoreBase using Supabase."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from vector_store.base import VectorStore as VectorStoreBase

from shared.exceptions import DocumentStoreError
from loggers import logger


class VectorStore(VectorStoreBase):
    """Manage vector operations in Supabase pgvector."""

    def __init__(self, supabase_client):
        self.client = supabase_client
        logger.info("VectorStore initialized")

    def find_document_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Look up a document by its content_hash.

        Args:
            content_hash: SHA-256 hex digest of the file bytes.

        Returns:
            Document dict or None if no match found.
        """
        try:
            response = (
                self.client.table("documents")
                .select("id, filename, content_hash, version_date")
                .eq("content_hash", content_hash)
                .order("version_date", desc=True)
                .limit(1)
                .execute()
            )
            results = response.data or []
            return results[0] if results else None
        except Exception as e:
            raise DocumentStoreError(
                message=f"Failed to find document by hash: {str(e)}",
                error_code="DOCUMENT_HASH_LOOKUP_ERROR",
                details={"content_hash": content_hash},
            )

    def insert_document(
        self,
        filename: str,
        version_date: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        content_hash: Optional[str] = None,
    ) -> str:
        try:
            version_date = (
                version_date.isoformat()
                if isinstance(version_date, datetime)
                else (version_date or datetime.now(timezone.utc).isoformat())
            )
            doc_id = str(uuid.uuid4())
            metadata = metadata or {}

            insert_data = {
                "id": doc_id,
                "filename": filename,
                "version_date": version_date,
                "metadata": metadata,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            if content_hash is not None:
                insert_data["content_hash"] = content_hash

            self.client.table("documents").insert(insert_data).execute()

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
                    # Supabase REST API (PostgREST) needs the vector as a
                    # string representation "[0.1, 0.2, ...]" so it can cast
                    # it to the vector(1536) column type. A raw Python list
                    # gets serialised as a JSON array and is NOT auto-cast.
                    "embedding": str(chunk["embedding"]),
                    "metadata": chunk.get("metadata", {}),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                for idx, chunk in enumerate(chunks)
            ]

            self.client.table("document_chunks").insert(chunk_records).execute()
            logger.info(
                f"Inserted {len(chunk_records)} chunks for document {document_id}"
            )
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
        latest_only: bool = False,
    ) -> List[Dict[str, Any]]:
        try:
            rpc_params = {
                # String representation so PostgREST casts to vector(1536).
                "query_embedding": str(query_embedding),
                "top_k": top_k,
                "latest_only": latest_only,
            }
            if version_filter is not None:
                rpc_params["version_filter"] = (
                    version_filter.isoformat()
                    if isinstance(version_filter, datetime)
                    else version_filter
                )

            response = self.client.rpc("search_similar_chunks", rpc_params).execute()

            results = response.data or []
            # Map out_* column names back to standard names
            mapped_results = []
            for r in results:
                mapped_results.append(
                    {
                        "id": r.get("out_id"),
                        "document_id": r.get("out_document_id"),
                        "text": r.get("out_text"),
                        "chunk_index": r.get("out_chunk_index"),
                        "metadata": r.get("out_metadata"),
                        "filename": r.get("out_filename"),
                        "version_date": r.get("out_version_date"),
                        "similarity_score": r.get("out_similarity_score"),
                    }
                )
            results = mapped_results

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
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                }
            ).execute()

            logger.info(
                f"Logged ingestion: {filename} ({status}, {chunk_count} chunks)"
            )
        except Exception as e:
            raise DocumentStoreError(
                message=f"Failed to log ingestion: {str(e)}",
                error_code="INGESTION_LOG_ERROR",
            )

    async def health_check(self) -> bool:
        try:
            response = (
                self.client.table("documents")
                .select("id", count="exact")
                .limit(1)
                .execute()
            )
            logger.info("Health check passed: database is healthy")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False

    # ── AI Decision persistence ────────────────────────────────────────

    def insert_ai_decisions(self, records: List[Dict[str, Any]]) -> int:
        """Insert multiple AI decision records.

        Args:
            records: List of row dicts matching the ai_decisions table schema.

        Returns:
            Number of records successfully inserted.
        """
        if not records:
            return 0

        try:
            self.client.table("ai_decisions").insert(records).execute()
            return len(records)
        except Exception as e:
            logger.warning(f"VectorStore: AI decision batch insert failed: {e}")
            return 0

    def load_ai_decisions(self, limit: int) -> List[Dict[str, Any]]:
        """Load the most recent AI decision records, oldest first.

        Args:
            limit: Maximum number of records to load.

        Returns:
            List of row dicts ordered by timestamp DESC, reversed to oldest first.
        """
        try:
            response = (
                self.client.table("ai_decisions")
                .select("*")
                .order("timestamp", desc=True)
                .limit(limit)
                .execute()
            )
            rows = response.data or []
            rows.reverse()  # Oldest first (matches deque append order)
            return rows
        except Exception as e:
            logger.warning(f"VectorStore: AI decision load failed: {e}")
            return []

    def update_ai_decision_feedback(self, run_id: str, feedback: Dict[str, Any]) -> bool:
        """Update user feedback for a specific AI decision run_id.

        Args:
            run_id: LangSmith run ID.
            feedback: Feedback dict (e.g. {"score": 1, "comment": "great"}).

        Returns:
            True on success, False on failure.
        """
        try:
            self.client.table("ai_decisions").update(
                {"user_feedback": feedback}
            ).eq("run_id", run_id).execute()
            return True
        except Exception as e:
            logger.warning(f"VectorStore: AI decision feedback update failed: {e}")
            return False
