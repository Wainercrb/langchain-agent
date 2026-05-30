"""Unit tests for document formatting utilities."""

import pytest

from utils.formatting import format_documents_as_context
from models.retrieval import RetrievedDocument


class TestFormatDocumentsAsContext:
    """Tests for format_documents_as_context function."""

    def test_format_single_document(self):
        """Should format a single document correctly."""
        docs = [
            RetrievedDocument(
                document_id="doc-1",
                chunk_id="chunk-1",
                filename="test.txt",
                text="This is test content.",
                similarity_score=0.95,
            )
        ]
        
        result = format_documents_as_context(docs)
        
        assert "[Document 1]" in result
        assert "test.txt" in result
        assert "95.00%" in result
        assert "This is test content." in result

    def test_format_multiple_documents(self, sample_documents):
        """Should format multiple documents with correct numbering."""
        result = format_documents_as_context(sample_documents)
        
        assert "[Document 1]" in result
        assert "[Document 2]" in result
        assert "[Document 3]" in result
        assert "test1.txt" in result
        assert "test2.txt" in result
        assert "test3.txt" in result

    def test_format_empty_list_default_message(self):
        """Should return default message for empty list."""
        result = format_documents_as_context([])
        
        assert result == "No context documents available."

    def test_format_empty_list_custom_message(self):
        """Should return custom message for empty list."""
        custom_msg = "No documents found."
        result = format_documents_as_context([], empty_message=custom_msg)
        
        assert result == custom_msg

    def test_format_similarity_percentage(self):
        """Should format similarity score as percentage."""
        docs = [
            RetrievedDocument(
                document_id="doc-1",
                chunk_id="chunk-1",
                filename="test.txt",
                text="Content",
                similarity_score=0.875,
            )
        ]
        
        result = format_documents_as_context(docs)
        
        assert "87.50%" in result

    def test_format_preserves_content(self):
        """Should preserve document content exactly."""
        content = "Line 1\nLine 2\nLine 3"
        docs = [
            RetrievedDocument(
                document_id="doc-1",
                chunk_id="chunk-1",
                filename="test.txt",
                text=content,
                similarity_score=0.90,
            )
        ]
        
        result = format_documents_as_context(docs)
        
        assert content in result

    def test_format_long_filename(self):
        """Should handle long filenames."""
        docs = [
            RetrievedDocument(
                document_id="doc-1",
                chunk_id="chunk-1",
                filename="very_long_filename_that_might_cause_issues_in_formatting.pdf",
                text="Content",
                similarity_score=0.90,
            )
        ]
        
        result = format_documents_as_context(docs)
        
        assert "very_long_filename" in result
