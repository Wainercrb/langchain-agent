"""Unit tests for similarity threshold filtering."""

import pytest

from domain.utils.filters import filter_by_threshold


class TestFilterByThreshold:
    """Tests for filter_by_threshold function."""

    @pytest.fixture
    def sample_results(self):
        """Sample search results as dictionaries (from vector store)."""
        return [
            {
                "document_id": "doc-1",
                "id": "chunk-1",
                "text": "First document about Python.",
                "similarity_score": 0.95,
                "filename": "test1.txt",
            },
            {
                "document_id": "doc-2",
                "id": "chunk-2",
                "text": "Second document about ML.",
                "similarity_score": 0.85,
                "filename": "test2.txt",
            },
            {
                "document_id": "doc-3",
                "id": "chunk-3",
                "text": "Third document about data science.",
                "similarity_score": 0.75,
                "filename": "test3.txt",
            },
        ]

    def test_filter_above_threshold(self, sample_results):
        """Should keep documents above threshold."""
        filtered = list(filter_by_threshold(sample_results, threshold=0.80))
        
        assert len(filtered) == 2
        assert all(doc["similarity_score"] >= 0.80 for doc in filtered)

    def test_filter_below_threshold(self, sample_results):
        """Should remove documents below threshold."""
        filtered = list(filter_by_threshold(sample_results, threshold=0.90))
        
        assert len(filtered) == 1
        assert filtered[0]["similarity_score"] >= 0.90

    def test_filter_all_above_threshold(self, sample_results):
        """Should keep all documents if all are above threshold."""
        filtered = list(filter_by_threshold(sample_results, threshold=0.70))
        
        assert len(filtered) == 3

    def test_filter_all_below_threshold(self, sample_results):
        """Should return empty list if all are below threshold."""
        filtered = list(filter_by_threshold(sample_results, threshold=0.99))
        
        assert len(filtered) == 0

    def test_filter_empty_list(self):
        """Should handle empty document list."""
        filtered = list(filter_by_threshold([], threshold=0.80))
        
        assert filtered == []

    def test_filter_exact_threshold(self):
        """Should include documents at exactly the threshold."""
        docs = [
            {
                "document_id": "exact",
                "id": "chunk-exact",
                "text": "Exact match",
                "similarity_score": 0.85,
            }
        ]
        
        filtered = list(filter_by_threshold(docs, threshold=0.85))
        
        assert len(filtered) == 1
        assert filtered[0]["similarity_score"] == 0.85

    def test_filter_preserves_order(self, sample_results):
        """Should preserve original document order."""
        filtered = list(filter_by_threshold(sample_results, threshold=0.70))
        
        # Check order is preserved
        scores = [doc["similarity_score"] for doc in filtered]
        assert scores == sorted(scores, reverse=True)
