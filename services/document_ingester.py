"""Document ingestion orchestrator coordinating all RAG components."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class DocumentIngester:
    """Orchestrate document ingestion: parse -> chunk -> embed -> store."""

    def __init__(
        self,
        document_processor,
        embeddings_wrapper,
        vector_store,
        version_manager,
        state_tracker,
    ):
        """
        Initialize document ingester with all dependencies.

        Args:
            document_processor: DocumentProcessor instance
            embeddings_wrapper: GoogleEmbeddingsWrapper instance
            vector_store: VectorStore instance
            version_manager: VersionManager instance
            state_tracker: StateTracker instance
        """
        self.processor = document_processor
        self.embeddings = embeddings_wrapper
        self.vector_store = vector_store
        self.version_manager = version_manager
        self.state_tracker = state_tracker
        logger.info("DocumentIngester initialized")

    def ingest_file(
        self,
        file_path: Path,
        md5_hash: str,
        custom_version_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a single file end-to-end.

        Args:
            file_path: Path to file to ingest
            md5_hash: MD5 hash of file for tracking
            custom_version_date: Optional custom version date

        Returns:
            Ingestion result dict with:
            - status: 'success' or 'error'
            - document_id: UUID of inserted document
            - chunk_count: Number of chunks created
            - version_date: Version date used
            - error_message: Error message if failed

        Raises:
            Exception: If critical error occurs
        """
        from utils.exceptions import IngestionError

        result = {
            "filename": file_path.name,
            "status": "error",
            "document_id": None,
            "chunk_count": 0,
            "version_date": None,
            "error_message": None,
        }

        try:
            logger.info(f"Starting ingestion: {file_path.name}")

            # Step 1: Parse file
            logger.debug(f"Parsing file: {file_path.name}")
            text_content = self.processor.parse_file(file_path)
            logger.info(f"Parsed file: {len(text_content)} characters")

            # Step 2: Chunk text
            logger.debug(f"Chunking text: {file_path.name}")
            chunks = self.processor.chunk_text(text_content)
            logger.info(f"Created {len(chunks)} chunks")

            # Step 3: Generate version date
            version_date = self.version_manager.generate_version_date(custom_version_date)

            # Step 4: Embed chunks (with embeddings)
            logger.debug(f"Generating embeddings: {len(chunks)} chunks")
            embeddings = self.embeddings.embed_documents([chunk for chunk in chunks])
            logger.info(f"Generated {len(embeddings)} embeddings")

            # Step 5: Prepare chunk records with embeddings
            chunk_records = [
                {
                    "text": chunk,
                    "embedding": embedding,
                    "metadata": {"chunk_index": idx, "char_count": len(chunk)},
                }
                for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ]

            # Step 6: Insert document into Supabase
            logger.debug(f"Inserting document: {file_path.name}")
            document_id = self.vector_store.insert_document(
                filename=file_path.name,
                version_date=version_date,
                metadata={
                    "source": "batch_ingestion",
                    "file_size": file_path.stat().st_size,
                    "chunk_count": len(chunks),
                },
            )
            logger.info(f"Inserted document: {document_id}")

            # Step 7: Insert chunks
            logger.debug(f"Inserting {len(chunk_records)} chunks")
            chunk_count = self.vector_store.insert_chunks(document_id, chunk_records)
            logger.info(f"Inserted {chunk_count} chunks")

            # Step 8: Update tracking state
            self.state_tracker.mark_processed(
                filename=file_path.name,
                md5_hash=md5_hash,
                version_date=version_date,
                chunk_count=chunk_count,
                document_id=document_id,
            )

            # Step 9: Log successful ingestion
            self.vector_store.log_ingestion(
                filename=file_path.name,
                status="success",
                chunk_count=chunk_count,
            )

            result.update(
                {
                    "status": "success",
                    "document_id": document_id,
                    "chunk_count": chunk_count,
                    "version_date": version_date.isoformat(),
                }
            )

            logger.info(
                f"✅ Ingestion successful: {file_path.name} ({chunk_count} chunks, doc_id={document_id[:8]}...)"
            )
            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Ingestion failed for {file_path.name}: {error_msg}")

            # Log failure
            try:
                self.vector_store.log_ingestion(
                    filename=file_path.name,
                    status="failure",
                    error_message=error_msg,
                )
            except Exception as log_e:
                logger.error(f"Failed to log ingestion error: {str(log_e)}")

            result.update(
                {
                    "status": "error",
                    "error_message": error_msg,
                }
            )

            return result

    def ingest_batch(self, files: list[Path], file_hashes: Dict[str, str]) -> Dict[str, Any]:
        """
        Ingest multiple files in batch.

        Args:
            files: List of file paths to ingest
            file_hashes: Dict mapping filename -> md5_hash

        Returns:
            Batch result summary with success/failure counts
        """
        results = {
            "total_files": len(files),
            "successful": 0,
            "failed": 0,
            "ingested_chunks": 0,
            "errors": [],
            "details": [],
        }

        for file_path in files:
            try:
                md5_hash = file_hashes.get(file_path.name, "unknown")
                result = self.ingest_file(file_path, md5_hash)
                results["details"].append(result)

                if result["status"] == "success":
                    results["successful"] += 1
                    results["ingested_chunks"] += result["chunk_count"]
                else:
                    results["failed"] += 1
                    results["errors"].append({"file": file_path.name, "error": result["error_message"]})

            except Exception as e:
                logger.error(f"Batch ingestion failed for {file_path.name}: {str(e)}")
                results["failed"] += 1
                results["errors"].append({"file": file_path.name, "error": str(e)})

        logger.info(
            f"Batch ingestion summary: {results['successful']} successful, "
            f"{results['failed']} failed, {results['ingested_chunks']} total chunks"
        )

        return results
