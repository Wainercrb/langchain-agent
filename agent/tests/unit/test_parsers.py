"""Unit tests for file parsers."""

import pytest
from pathlib import Path

from infrastructure.parsers.parser import (
    TextParser,
    MarkdownParser,
    CSVParser,
    ParserFactory,
)


class TestTextParser:
    """Tests for TextParser."""

    def test_parse_txt_file(self, temp_txt_file: Path):
        """Should parse a text file and return its content."""
        parser = TextParser()
        result = parser.parse(temp_txt_file)
        
        assert result is not None
        assert "test document" in result.lower()
        assert "multiple lines" in result.lower()

    def test_parse_empty_file(self, tmp_path: Path):
        """Should handle empty files gracefully."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        
        parser = TextParser()
        result = parser.parse(empty_file)
        
        assert result == ""

    def test_parse_nonexistent_file(self, tmp_path: Path):
        """Should raise error for non-existent file."""
        parser = TextParser()
        nonexistent = tmp_path / "does_not_exist.txt"
        
        with pytest.raises(FileNotFoundError):
            parser.parse(nonexistent)


class TestMarkdownParser:
    """Tests for MarkdownParser."""

    def test_parse_markdown_file(self, temp_md_file: Path):
        """Should parse markdown and extract text content."""
        parser = MarkdownParser()
        result = parser.parse(temp_md_file)
        
        assert result is not None
        assert "Test Document" in result or "test document" in result.lower()
        assert "markdown" in result.lower()

    def test_parse_markdown_with_lists(self, tmp_path: Path):
        """Should handle markdown lists."""
        md_file = tmp_path / "list.md"
        md_file.write_text("# List\n\n- Item 1\n- Item 2\n- Item 3")
        
        parser = MarkdownParser()
        result = parser.parse(md_file)
        
        assert "Item 1" in result
        assert "Item 2" in result


class TestCSVParser:
    """Tests for CSVParser."""

    def test_parse_csv_file(self, temp_csv_file: Path):
        """Should parse CSV and return formatted text."""
        parser = CSVParser()
        result = parser.parse(temp_csv_file)
        
        assert result is not None
        assert "Alice" in result
        assert "Bob" in result

    def test_parse_csv_with_headers(self, tmp_path: Path):
        """Should include headers in output."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,value\nTest,123\nAnother,456")
        
        parser = CSVParser()
        result = parser.parse(csv_file)
        
        assert "name" in result or "Test" in result


class TestParserFactory:
    """Tests for ParserFactory."""

    def test_get_parser_for_txt(self):
        """Should return TextParser for .txt files."""
        parser = ParserFactory.get_parser(".txt")
        assert isinstance(parser, TextParser)

    def test_get_parser_for_md(self):
        """Should return MarkdownParser for .md files."""
        parser = ParserFactory.get_parser(".md")
        assert isinstance(parser, MarkdownParser)

    def test_get_parser_for_csv(self):
        """Should return CSVParser for .csv files."""
        parser = ParserFactory.get_parser(".csv")
        assert isinstance(parser, CSVParser)

    def test_get_parser_unsupported_format(self):
        """Should raise error for unsupported formats."""
        from utils.exceptions import IngestionError
        with pytest.raises(IngestionError, match="Unsupported"):
            ParserFactory.get_parser(".xyz")

    def test_get_parser_case_insensitive(self):
        """Should handle uppercase extensions."""
        parser = ParserFactory.get_parser(".TXT")
        assert isinstance(parser, TextParser)
