"""Document ingestion pipeline — orchestrates parsing, splitting, embedding, and storage."""

import hashlib
import shutil
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Protocol

from langchain_text_splitters import RecursiveCharacterTextSplitter

from loggers.base import Logger
from ingestion.parsers.parser import ParserFactory as ParserRegistry


class IngestionStatus(str, Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class IngestionResult:
    filename: str
    status: IngestionStatus
    chunk_count: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None


class EmbeddingsProvider(Protocol):
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...


class VectorStoreProvider(Protocol):
    def find_document_by_hash(self, content_hash: str) -> dict[str, Any] | None: ...
    def insert_document(self, filename: str, **kwargs) -> str: ...
    def insert_chunks(self, document_id: str, chunks: List[dict[str, Any]]) -> int: ...
    def log_ingestion(
        self,
        filename: str,
        status: str,
        chunk_count: int = 0,
        error_message: str | None = None,
    ) -> None: ...


class DocumentIngestionPipeline:
    """Processes documents through parsing, splitting, embedding, and storage.

    Accepts dependencies via constructor injection for testability.
    """

    def __init__(
        self,
        embeddings: EmbeddingsProvider,
        vector_store: VectorStoreProvider,
        processed_dir: Path,
        failed_dir: Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        logger: Logger = None,
        parser_registry: ParserRegistry = None,
    ):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.processed_dir = processed_dir
        self.failed_dir = failed_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logger
        self._parser_registry = parser_registry

    def ingest_file(self, file_path: Path) -> IngestionResult:
        """Process a single file through the ingestion pipeline."""
        start = time.time()
        if self.logger:
            self.logger.info(f"Processing: {file_path.name}")

        raw_bytes = file_path.read_bytes()
        content_hash = hashlib.sha256(raw_bytes).hexdigest()

        existing = self.vector_store.find_document_by_hash(content_hash)
        if existing is not None:
            if self.logger:
                self.logger.info(f"Skipping (unchanged): {file_path.name}")
            self._move_to_processed(file_path)
            return IngestionResult(
                filename=file_path.name,
                status=IngestionStatus.SKIPPED,
                elapsed_seconds=time.time() - start,
            )

        try:
            text = self._parse_file(file_path)
        except Exception as e:
            return self._handle_failure(file_path, str(e), start)

        chunks = self._split_text(text)
        if not chunks:
            return self._handle_failure(
                file_path, "No content after splitting", start
            )

        try:
            vectors = self.embeddings.embed_documents(chunks)
        except Exception as e:
            return self._handle_failure(file_path, str(e), start)

        doc_id = self.vector_store.insert_document(
            file_path.name, content_hash=content_hash
        )
        chunk_records = [
            {"text": chunks[i], "embedding": vectors[i]} for i in range(len(chunks))
        ]
        count = self.vector_store.insert_chunks(doc_id, chunk_records)
        self.vector_store.log_ingestion(file_path.name, "success", chunk_count=count)

        self._move_to_processed(file_path)

        elapsed = time.time() - start
        if self.logger:
            self.logger.info(f"Done: {file_path.name} ({count} chunks, {elapsed:.1f}s)")
        return IngestionResult(
            filename=file_path.name,
            status=IngestionStatus.SUCCESS,
            chunk_count=count,
            elapsed_seconds=elapsed,
        )

    def ingest_directory(self, directory: Path) -> List[IngestionResult]:
        """Discover and process all files in a directory."""
        files = sorted(
            [
                f
                for f in directory.iterdir()
                if f.is_file() and not f.name.startswith(".")
            ]
        )

        if not files:
            return []

        if self.logger:
            self.logger.info(f"Found {len(files)} file(s) in {directory}")
        return [self.ingest_file(f) for f in files]

    def _parse_file(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        parser = self._parser_registry.get_parser(ext)
        return parser.parse(file_path)

    def _split_text(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_text(text)
        return [c for c in chunks if c.strip()]

    def _handle_failure(
        self, file_path: Path, error: str, start: float
    ) -> IngestionResult:
        if self.logger:
            self.logger.error(f"Failed: {file_path.name} — {error}")
        self._move_to_failed(file_path)
        self.vector_store.log_ingestion(
            file_path.name, "failure", error_message=error
        )
        return IngestionResult(
            filename=file_path.name,
            status=IngestionStatus.FAILED,
            elapsed_seconds=time.time() - start,
            error=error,
        )

    def _move_to_processed(self, file_path: Path) -> None:
        shutil.move(str(file_path), str(self.processed_dir / file_path.name))

    def _move_to_failed(self, file_path: Path) -> None:
        shutil.move(str(file_path), str(self.failed_dir / file_path.name))

    def retry_failed_files(self, max_retries: int = 3) -> List[IngestionResult]:
        """Re-process files in the failed/ directory with retry tracking.

        Files that have exceeded max_retries are moved to failed/permanent/
        and logged as unrecoverable.

        Args:
            max_retries: Maximum retry attempts before giving up.

        Returns:
            List of IngestionResult for each retry attempt.
        """
        permanent_dir = self.failed_dir / "permanent"
        permanent_dir.mkdir(parents=True, exist_ok=True)

        retry_meta_dir = self.failed_dir / ".retry_meta"
        retry_meta_dir.mkdir(parents=True, exist_ok=True)

        failed_files = [
            f for f in self.failed_dir.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]

        if not failed_files:
            return []

        results: List[IngestionResult] = []
        if self.logger:
            self.logger.info(f"DLQ retry: {len(failed_files)} file(s) in failed/")

        for file_path in failed_files:
            meta_path = retry_meta_dir / f"{file_path.name}.json"
            retry_count = 0

            # Read retry count from metadata
            if meta_path.exists():
                import json
                try:
                    meta = json.loads(meta_path.read_text())
                    retry_count = meta.get("retries", 0)
                except Exception:
                    retry_count = 0

            if retry_count >= max_retries:
                # Move to permanent failure
                shutil.move(str(file_path), str(permanent_dir / file_path.name))
                if meta_path.exists():
                    meta_path.unlink()
                if self.logger:
                    self.logger.warning(
                        f"DLQ: {file_path.name} exceeded {max_retries} retries, "
                        f"moved to permanent/"
                    )
                results.append(IngestionResult(
                    filename=file_path.name,
                    status=IngestionStatus.FAILED,
                    error=f"Exceeded {max_retries} retries",
                ))
                continue

            # Attempt retry
            if self.logger:
                self.logger.info(
                    f"DLQ retry ({retry_count + 1}/{max_retries}): {file_path.name}"
                )

            # Temporarily move back to knowledge dir for processing
            temp_path = self.failed_dir.parent / "raw_docs" / f".retry_{file_path.name}"
            shutil.copy(str(file_path), str(temp_path))

            result = self.ingest_file(temp_path)

            # Update retry metadata if still failed
            if result.status == IngestionStatus.FAILED:
                import json
                new_count = retry_count + 1
                meta_path.write_text(json.dumps({
                    "filename": file_path.name,
                    "retries": new_count,
                    "last_error": result.error or "Unknown",
                    "last_attempt": time.time(),
                }))
                # Re-move to failed/ (ingest_file moves it there again)
                if self.logger:
                    self.logger.warning(
                        f"DLQ retry failed for {file_path.name}: {result.error}"
                    )
            else:
                # Clean up metadata on success
                if meta_path.exists():
                    meta_path.unlink()
                if self.logger:
                    self.logger.info(f"DLQ retry succeeded for {file_path.name}")

            results.append(result)

        return results
