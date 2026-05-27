"""Document parsing and text chunking for embeddings."""

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process documents: parse, chunk, validate."""

    def __init__(self):
        """Initialize document processor with RecursiveCharacterTextSplitter."""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from config import settings

            self.settings = settings
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
        """
        Parse document file to extract text.

        Args:
            file_path: Path to document file

        Returns:
            Extracted text content

        Raises:
            Exception: If parsing fails
        """
        from utils.exceptions import IngestionError

        try:
            extension = file_path.suffix.lower()

            if extension == ".txt":
                return self._parse_txt(file_path)
            elif extension == ".md":
                return self._parse_markdown(file_path)
            elif extension == ".html":
                return self._parse_html(file_path)
            elif extension == ".pdf":
                return self._parse_pdf(file_path)
            elif extension == ".docx":
                return self._parse_docx(file_path)
            elif extension == ".csv":
                return self._parse_csv(file_path)
            else:
                raise IngestionError(
                    message=f"Unsupported file type: {extension}",
                    error_code="UNSUPPORTED_FILE_TYPE",
                )
        except IngestionError:
            raise
        except Exception as e:
            logger.error(f"Failed to parse file {file_path}: {str(e)}")
            raise IngestionError(
                message=f"Failed to parse file {file_path}: {str(e)}",
                error_code="FILE_PARSE_ERROR",
                details={"file": str(file_path), "error": str(e)},
            )

    def _parse_txt(self, file_path: Path) -> str:
        """Parse plain text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        logger.info(f"Parsed .txt file: {file_path.name} ({len(text)} chars)")
        return text

    def _parse_markdown(self, file_path: Path) -> str:
        """Parse markdown file."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        logger.info(f"Parsed .md file: {file_path.name} ({len(text)} chars)")
        return text

    def _parse_html(self, file_path: Path) -> str:
        """Parse HTML file to extract text content."""
        try:
            from html.parser import HTMLParser
            import re

            class HTMLTextExtractor(HTMLParser):
                """Extract text from HTML, ignoring scripts and styles."""
                def __init__(self):
                    super().__init__()
                    self.text_parts = []
                    self.skip_content = False

                def handle_starttag(self, tag, attrs):
                    if tag in ['script', 'style']:
                        self.skip_content = True

                def handle_endtag(self, tag):
                    if tag in ['script', 'style']:
                        self.skip_content = False
                    elif tag in ['p', 'div', 'br', 'li']:
                        self.text_parts.append('\n')

                def handle_data(self, data):
                    if not self.skip_content:
                        text = data.strip()
                        if text:
                            self.text_parts.append(text)

            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            parser = HTMLTextExtractor()
            parser.feed(html_content)
            text = '\n'.join(parser.text_parts)
            # Clean up excessive whitespace
            text = re.sub(r'\n\s*\n', '\n\n', text).strip()

            logger.info(f"Parsed .html file: {file_path.name} ({len(text)} chars)")
            return text
        except Exception as e:
            logger.error(f"Failed to parse HTML file {file_path}: {str(e)}")
            raise IngestionError(
                message=f"Failed to parse HTML file {file_path}: {str(e)}",
                error_code="HTML_PARSE_ERROR",
            )

    def _parse_pdf(self, file_path: Path) -> str:
        """Parse PDF file using pdfplumber."""
        try:
            import pdfplumber

            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    text += page_text or ""
            logger.info(f"Parsed .pdf file: {file_path.name} ({len(text)} chars, {len(pdf.pages)} pages)")
            return text
        except ImportError:
            from utils.exceptions import IngestionError

            raise IngestionError(
                message="pdfplumber not installed. Run: pip install pdfplumber",
                error_code="DEPENDENCY_MISSING",
            )

    def _parse_docx(self, file_path: Path) -> str:
        """Parse DOCX file using python-docx."""
        try:
            from docx import Document

            doc = Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
            logger.info(f"Parsed .docx file: {file_path.name} ({len(text)} chars)")
            return text
        except ImportError:
            from utils.exceptions import IngestionError

            raise IngestionError(
                message="python-docx not installed. Run: pip install python-docx",
                error_code="DEPENDENCY_MISSING",
            )

    def _parse_csv(self, file_path: Path) -> str:
        """Parse CSV file using pandas."""
        try:
            import pandas as pd

            df = pd.read_csv(file_path)
            text = df.to_string()
            logger.info(f"Parsed .csv file: {file_path.name} ({len(text)} chars, {len(df)} rows)")
            return text
        except ImportError:
            from utils.exceptions import IngestionError

            raise IngestionError(
                message="pandas not installed. Run: pip install pandas",
                error_code="DEPENDENCY_MISSING",
            )

    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text using RecursiveCharacterTextSplitter.

        Args:
            text: Full text to chunk

        Returns:
            List of text chunks

        Raises:
            Exception: If chunking fails
        """
        from utils.exceptions import IngestionError

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
