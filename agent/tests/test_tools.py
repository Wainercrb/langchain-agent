"""Unit tests for the tool functions.

Tests each tool independently with various inputs to verify:
- Web search: with mocked DuckDuckGo wrapper
- Search documents: with mocked retriever
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from services.tools.search_documents import _format_context, create_search_documents_tool
from services.tools.web_search import _format_search_results
from models import RetrievedDocument


# ═══════════════════════════════════════════════════════════════════════
#  Search Documents Tool Tests
# ═══════════════════════════════════════════════════════════════════════


class TestSearchDocumentsTool:
    """Tests for the search_documents tool."""

    def test_format_context_with_docs(self):
        docs = [
            RetrievedDocument(
                document_id="doc1",
                chunk_id="chunk1",
                text="Content of document one.",
                similarity_score=0.95,
                filename="doc1.pdf",
                version_date=datetime(2025, 1, 1),
            ),
            RetrievedDocument(
                document_id="doc2",
                chunk_id="chunk2",
                text="Content of document two.",
                similarity_score=0.82,
                filename="doc2.txt",
                version_date=datetime(2025, 2, 1),
            ),
        ]
        result = _format_context(docs)
        assert "doc1.pdf" in result
        assert "doc2.txt" in result
        assert "95.00%" in result
        assert "82.00%" in result
        assert "Content of document one." in result

    def test_format_context_empty(self):
        result = _format_context([])
        assert result == "No relevant documents found."

    def test_create_tool_with_retriever(self):
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            RetrievedDocument(
                document_id="doc1",
                chunk_id="chunk1",
                text="Test content.",
                similarity_score=0.9,
                filename="test.pdf",
                version_date=datetime(2025, 1, 1),
            )
        ]

        artifact_store = []
        tool = create_search_documents_tool(
            retriever=mock_retriever,
            artifact_store=artifact_store,
        )

        assert tool.name == "search_documents"
        assert "document knowledge base" in tool.description.lower()

        # Invoke the tool
        result = tool.invoke({"query": "test query", "top_k": 3})
        assert "test.pdf" in result
        assert "90.00%" in result
        assert len(artifact_store) == 1
        assert artifact_store[0].document_id == "doc1"

    def test_tool_error_handling(self):
        mock_retriever = Mock()
        mock_retriever.retrieve.side_effect = Exception("Retriever failure")

        artifact_store = []
        tool = create_search_documents_tool(
            retriever=mock_retriever,
            artifact_store=artifact_store,
        )

        result = tool.invoke({"query": "test", "top_k": 5})
        assert "Error" in result
        assert len(artifact_store) == 0

    def test_tool_without_artifact_store(self):
        """Tool should work even without an artifact store."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            RetrievedDocument(
                document_id="doc1",
                chunk_id="chunk1",
                text="Test content.",
                similarity_score=0.9,
                filename="test.pdf",
                version_date=datetime(2025, 1, 1),
            )
        ]

        tool = create_search_documents_tool(retriever=mock_retriever)
        result = tool.invoke({"query": "test", "top_k": 3})
        assert "test.pdf" in result


# ═══════════════════════════════════════════════════════════════════════
#  Web Search Tool Tests
# ═══════════════════════════════════════════════════════════════════════


class TestWebSearchTool:
    """Tests for the web search tool."""

    def test_format_search_results_with_data(self):
        results = [
            {"title": "Result 1", "snippet": "Snippet 1", "link": "https://example.com/1"},
            {"title": "Result 2", "snippet": "Snippet 2", "link": "https://example.com/2"},
        ]
        formatted = _format_search_results(results)
        assert "[Result 1]" in formatted
        assert "Result 1" in formatted
        assert "Snippet 1" in formatted
        assert "https://example.com/1" in formatted

    def test_format_search_results_empty(self):
        formatted = _format_search_results([])
        assert formatted == "No search results found."

    def test_format_search_results_truncates_to_three(self):
        results = [
            {"title": f"Result {i}", "snippet": f"Snippet {i}", "link": f"https://example.com/{i}"}
            for i in range(1, 10)
        ]
        formatted = _format_search_results(results)
        # Should only have 3 results
        assert formatted.count("[Result ") == 3

    @patch("ddgs.DDGS")
    def test_web_search_tool_success(self, mock_ddgs_class):
        mock_ddgs = MagicMock()
        mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs
        mock_ddgs.text.return_value = [
            {"title": "Test Title", "body": "Test snippet", "href": "https://example.com"},
        ]

        from services.tools.web_search import web_search_tool

        result = web_search_tool.invoke({"query": "test query"})
        assert "Test Title" in result
        assert "Test snippet" in result
        mock_ddgs.text.assert_called_once_with("test query", max_results=3)

    @patch("ddgs.DDGS")
    def test_web_search_tool_error(self, mock_ddgs_class):
        mock_ddgs = MagicMock()
        mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs
        mock_ddgs.text.side_effect = Exception("Search API error")

        from services.tools.web_search import web_search_tool

        result = web_search_tool.invoke({"query": "test"})
        assert "Error" in result
