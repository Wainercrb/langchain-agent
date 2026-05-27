"""Document parsing and text chunking for embeddings."""

import logging
from pathlib import Path
from typing import List

from utils.exceptions import IngestionError
from .parser import ParserFactory

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process documents: parse, chunk, validate."""

    def __init__(self):
        """Initialize document processor with RecursiveCharacterTextSplitter."""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            from config import settings

            self.settings = settings
            self._parser_factory = ParserFactory()
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
            )
            logger.info(
                f"DocumentProcessor initialized (chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap})"
            )
        except ImportError as e:
            logger.error(f"Failed to import RecursiveCharacterTextSplitter: {str(e)}")
            raise

    def parse_file(self, file_path: Path) -> str:
        try:
            extension = file_path.suffix.lower()
            parser = self._parser_factory.get_parser(extension)
            return parser.parse(file_path)

        except IngestionError:
            raise
        except Exception as e:
            logger.error(f"Failed to parse file {file_path}: {str(e)}")
            raise IngestionError(
                message=f"Failed to parse file {file_path}: {str(e)}",
                error_code="FILE_PARSE_ERROR",
                details={"file": str(file_path), "error": str(e)},
            )

    def register_custom_parser(self, parser):
        self._parser_factory.register_parser(parser)

    def chunk_text(self, text: str) -> List[str]:
        try:
            chunks = self.splitter.split_text(text)
            logger.info(f"Created {len(chunks)} chunks from text ({len(text)} chars)")
            return chunks
        except Exception as e:
            logger.error(f"Failed to chunk text: {str(e)}")
            raise IngestionError(
                message=f"Failed to chunk text: {str(e)}",
                error_code="CHUNKING_ERROR",
            )
