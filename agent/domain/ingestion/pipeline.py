"""Document ingestion pipeline — orchestrates parsing, splitting, embedding, and storage."""

import hashlib
import shutil
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Protocol

from langchain_text_splitters import RecursiveCharacterTextSplitter

from infrastructure.logging import logger
from infrastructure.parsers.parser import ParserFactory


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
    ):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.processed_dir = processed_dir
        self.failed_dir = failed_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_file(self, file_path: Path) -> IngestionResult:
        """Process a single file through the ingestion pipeline."""
        start = time.time()
        logger.info(f"Processing: {file_path.name}")

        raw_bytes = file_path.read_bytes()
        content_hash = hashlib.sha256(raw_bytes).hexdigest()

        existing = self.vector_store.find_document_by_hash(content_hash)
        if existing is not None:
            logger.info(f"Skipping (unchanged): {file_path.name}")
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
        logger.info(f"Done: {file_path.name} ({count} chunks, {elapsed:.1f}s)")
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

        logger.info(f"Found {len(files)} file(s) in {directory}")
        return [self.ingest_file(f) for f in files]

    def _parse_file(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        parser = ParserFactory.get_parser(ext)
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
        logger.error(f"Failed: {file_path.name} — {error}")
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
