"""File parser implementations using Strategy pattern.

ParserFactory auto-discovers parser subclasses via __init_subclass__
registration, making the system open for extension without modification (OCP).
"""

import re
from abc import ABC, abstractmethod
from html.parser import HTMLParser as StdlibHTMLParser
from pathlib import Path
from typing import Dict, List

from loggers import logger
from shared.exceptions import IngestionError


class ParserFactory:
    """Factory for selecting appropriate file parser based on file type.

    Uses registry pattern: parsers are auto-registered via FileParser's
    __init_subclass__ hook. No manual list maintenance needed.
    """

    _registry: Dict[str, "FileParser"] = {}

    @classmethod
    def register(cls, parser: "FileParser") -> None:
        """Register a parser for all file extensions it supports.

        Args:
            parser: FileParser instance to register.
        """
        for ext in [".txt", ".md", ".html", ".pdf", ".docx", ".csv"]:
            if parser.supports(ext):
                cls._registry[ext] = parser

    @classmethod
    def get_parser(cls, file_extension: str) -> "FileParser":
        """Get parser for file extension.

        Args:
            file_extension: File extension (e.g., '.pdf')

        Returns:
            Appropriate FileParser instance

        Raises:
            IngestionError: If no parser found for extension
        """
        parser = cls._registry.get(file_extension.lower())
        if parser is None:
            raise IngestionError(
                message=f"Unsupported file type: {file_extension}",
                error_code="UNSUPPORTED_FILE_TYPE",
            )
        return parser


class FileParser(ABC):
    """Abstract base class for file parsers.

    Concrete subclasses are automatically registered with ParserFactory
    via __init_subclass__. Subclasses define supports() and parse().
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Auto-register concrete subclasses in the factory registry
        ParserFactory.register(cls())

    @abstractmethod
    def supports(self, extension: str) -> bool:
        """Check if parser supports file extension."""
        pass

    @abstractmethod
    def parse(self, file_path: Path) -> str:
        """Parse file and return text content."""
        pass


class TextParser(FileParser):
    """Parser for plain text files."""

    def supports(self, extension: str) -> bool:
        return extension.lower() == ".txt"

    def parse(self, file_path: Path) -> str:
        """Parse plain text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        logger.info(f"Parsed .txt file: {file_path.name} ({len(text)} chars)")
        return text


class MarkdownParser(FileParser):
    """Parser for markdown files."""

    def supports(self, extension: str) -> bool:
        return extension.lower() == ".md"

    def parse(self, file_path: Path) -> str:
        """Parse markdown file."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        logger.info(f"Parsed .md file: {file_path.name} ({len(text)} chars)")
        return text


class HTMLParser(FileParser):
    """Parser for HTML files."""

    class _HTMLTextExtractor(StdlibHTMLParser):
        """Extract text from HTML, ignoring scripts and styles."""

        def __init__(self):
            super().__init__()
            self.text_parts = []
            self.skip_content = False

        def handle_starttag(self, tag, attrs):
            if tag in ["script", "style"]:
                self.skip_content = True

        def handle_endtag(self, tag):
            if tag in ["script", "style"]:
                self.skip_content = False
            elif tag in ["p", "div", "br", "li"]:
                self.text_parts.append("\n")

        def handle_data(self, data):
            if not self.skip_content:
                text = data.strip()
                if text:
                    self.text_parts.append(text)

    def supports(self, extension: str) -> bool:
        return extension.lower() == ".html"

    def parse(self, file_path: Path) -> str:
        """Parse HTML file to extract text content."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            parser = self._HTMLTextExtractor()
            parser.feed(html_content)
            text = "\n".join(parser.text_parts)
            # Clean up excessive whitespace
            text = re.sub(r"\n\s*\n", "\n\n", text).strip()

            logger.info(f"Parsed .html file: {file_path.name} ({len(text)} chars)")
            return text
        except Exception as e:
            logger.error(f"Failed to parse HTML file {file_path}: {str(e)}")
            raise IngestionError(
                message=f"Failed to parse HTML file {file_path}: {str(e)}",
                error_code="HTML_PARSE_ERROR",
            )


class PDFParser(FileParser):
    """Parser for PDF files."""

    def supports(self, extension: str) -> bool:
        return extension.lower() == ".pdf"

    def parse(self, file_path: Path) -> str:
        """Parse PDF file using pdfplumber."""
        try:
            import pdfplumber

            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    text += page_text or ""
            logger.info(f"Parsed .pdf file: {file_path.name} ({len(text)} chars)")
            return text
        except ImportError:
            raise IngestionError(
                message="pdfplumber not installed. Install with: pip install pdfplumber",
                error_code="DEPENDENCY_ERROR",
            )
        except Exception as e:
            logger.error(f"Failed to parse PDF file {file_path}: {str(e)}")
            raise IngestionError(
                message=f"Failed to parse PDF file {file_path}: {str(e)}",
                error_code="PDF_PARSE_ERROR",
            )


class DocxParser(FileParser):
    """Parser for DOCX files."""

    def supports(self, extension: str) -> bool:
        return extension.lower() == ".docx"

    def parse(self, file_path: Path) -> str:
        """Parse DOCX file using python-docx."""
        try:
            from docx import Document

            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            logger.info(f"Parsed .docx file: {file_path.name} ({len(text)} chars)")
            return text
        except ImportError:
            raise IngestionError(
                message="python-docx not installed. Install with: pip install python-docx",
                error_code="DEPENDENCY_ERROR",
            )
        except Exception as e:
            logger.error(f"Failed to parse DOCX file {file_path}: {str(e)}")
            raise IngestionError(
                message=f"Failed to parse DOCX file {file_path}: {str(e)}",
                error_code="DOCX_PARSE_ERROR",
            )


class CSVParser(FileParser):
    """Parser for CSV files."""

    def supports(self, extension: str) -> bool:
        return extension.lower() == ".csv"

    def parse(self, file_path: Path) -> str:
        """Parse CSV file."""
        try:
            import csv

            rows = []
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    rows.append(" | ".join(row))
            text = "\n".join(rows)
            logger.info(f"Parsed .csv file: {file_path.name} ({len(text)} chars)")
            return text
        except Exception as e:
            logger.error(f"Failed to parse CSV file {file_path}: {str(e)}")
            raise IngestionError(
                message=f"Failed to parse CSV file {file_path}: {str(e)}",
                error_code="CSV_PARSE_ERROR",
            )
