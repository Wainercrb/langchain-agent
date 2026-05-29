"""Unit tests for Retriever service.

Tests retrieval logic in isolation using mocked VectorStore and Embeddings.
"""

from datetime import date, datetime
from unittest.mock import Mock

import pytest

from rag.retrieval.retriever import RetrievedDocument, Retriever


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore with sample search results."""
    store = Mock()
    store.search_similar.return_value = [
        {
            "document_id": "doc1",
            "chunk_id": "chunk1",
            "text": "How to enroll in the program: Step 1: Fill out the form",
            "similarity_score": 0.95,
            "filename": "enrollment.pdf",
            "version_date": datetime(2025, 1, 1),
        },
        {
            "document_id": "doc2",
            "chunk_id": "chunk2",
            "text": "Enrollment requirements include: valid ID, proof of address",
            "similarity_score": 0.87,
            "filename": "requirements.txt",
            "version_date": datetime(2025, 1, 2),
        },
        {
            "document_id": "doc3",
            "chunk_id": "chunk3",
            "text": "Some unrelated content about parking",
            "similarity_score": 0.42,
            "filename": "parking.txt",
            "version_date": datetime(2025, 1, 3),
        },
    ]
    return store


@pytest.fixture
def mock_embeddings():
    """Mock GoogleEmbeddingsWrapper that returns a fake embedding."""
    embeddings = Mock()
    embeddings.embed_query.return_value = [0.1] * 1536  # Google 1536-dim embeddings
    return embeddings


@pytest.fixture
def retriever(mock_vector_store, mock_embeddings):
    """Create Retriever instance with mocks."""
    return Retriever(mock_vector_store, mock_embeddings)


def test_retrieve_success(retriever, mock_vector_store, mock_embeddings):
    """Test successful document retrieval."""
    result = retriever.retrieve("how to enroll", top_k=5)

    # Verify calls
    mock_embeddings.embed_query.assert_called_once_with("how to enroll")
    mock_vector_store.search_similar.assert_called_once()

    # Verify results (doc3 filtered by default threshold of 0.5)
    assert len(result) == 2
    assert isinstance(result[0], RetrievedDocument)
    assert result[0].document_id == "doc1"
    assert result[0].similarity_score == 0.95
    assert result[1].similarity_score == 0.87


def test_retrieve_no_results(retriever, mock_vector_store):
    """Test retrieval returns empty list when no results found."""
    mock_vector_store.search_similar.return_value = []
    result = retriever.retrieve("nonexistent query")

    assert result == []
    assert isinstance(result, list)


def test_retrieve_with_threshold(retriever, mock_vector_store):
    """Test retrieval filters by similarity threshold."""
    result = retriever.retrieve("how to enroll", top_k=5, similarity_threshold=0.90)

    # Should filter out similarity < 0.90 (docs 2 and 3)
    assert len(result) == 1
    assert result[0].similarity_score == 0.95
    assert all(r.similarity_score >= 0.90 for r in result)


def test_retrieve_with_strict_threshold(retriever, mock_vector_store):
    """Test retrieval with very high threshold."""
    result = retriever.retrieve("query", top_k=5, similarity_threshold=0.95)

    # Only doc1 has similarity >= 0.95
    assert len(result) == 1
    assert result[0].document_id == "doc1"


def test_retrieve_with_version_filter(retriever, mock_vector_store):
    """Test retrieval filters by document version date."""
    result = retriever.retrieve("how to enroll", version_filter=date(2025, 1, 2))

    # Should include docs with version_date >= 2025-01-02 (docs 2 and 3)
    assert len(result) == 2
    assert result[0].document_id == "doc2"
    assert result[1].document_id == "doc3"
    assert all(r.version_date.date() >= date(2025, 1, 2) for r in result)


def test_retrieve_with_threshold_and_version_filter(retriever, mock_vector_store):
    """Test retrieval with both threshold and version filters."""
    result = retriever.retrieve(
        "query", top_k=5, similarity_threshold=0.85, version_filter=date(2025, 1, 2)
    )

    # Should include docs with similarity >= 0.85 AND version >= 2025-01-02
    # doc1: 0.95 >= 0.85 but 2025-01-01 < 2025-01-02 (filtered out)
    # doc2: 0.87 >= 0.85 and 2025-01-02 >= 2025-01-02 (included)
    # doc3: 0.42 < 0.85 (filtered out)
    assert len(result) == 1
    assert result[0].document_id == "doc2"


def test_retrieve_error_handling(retriever, mock_vector_store, mock_embeddings):
    """Test retriever raises exception when vector store fails."""
    mock_vector_store.search_similar.side_effect = Exception("DB connection error")

    with pytest.raises(Exception, match="DB connection error"):
        retriever.retrieve("query")


def test_retrieve_embedding_error(retriever, mock_embeddings):
    """Test retriever raises exception when embedding fails."""
    mock_embeddings.embed_query.side_effect = Exception("API error")

    with pytest.raises(Exception, match="API error"):
        retriever.retrieve("query")


def test_retrieve_returns_correct_model(retriever):
    """Test that retrieve returns RetrievedDocument Pydantic models."""
    result = retriever.retrieve("query")

    assert len(result) > 0
    for doc in result:
        assert isinstance(doc, RetrievedDocument)
        assert hasattr(doc, "document_id")
        assert hasattr(doc, "chunk_id")
        assert hasattr(doc, "text")
        assert hasattr(doc, "similarity_score")
        assert hasattr(doc, "filename")
        assert hasattr(doc, "version_date")


def test_retrieve_passes_latest_only_true(retriever, mock_vector_store):
    """Test that latest_only=True is passed to vector store search."""
    result = retriever.retrieve("query", latest_only=True)

    call_args = mock_vector_store.search_similar.call_args
    assert call_args[1]["latest_only"] is True


def test_retrieve_passes_latest_only_false(retriever, mock_vector_store):
    """Test that latest_only=False is passed to vector store search."""
    result = retriever.retrieve("query", latest_only=False)

    call_args = mock_vector_store.search_similar.call_args
    assert call_args[1]["latest_only"] is False


def test_retrieve_default_latest_only(retriever, mock_vector_store):
    """Test that latest_only defaults to False for backwards compat."""
    result = retriever.retrieve("query")

    call_args = mock_vector_store.search_similar.call_args
    assert call_args[1]["latest_only"] is False


def test_retrieve_ordered_by_similarity(retriever):
    """Test that retrieve returns results ordered by similarity (highest first)."""
    result = retriever.retrieve("query")

    similarities = [doc.similarity_score for doc in result]
    assert similarities == sorted(similarities, reverse=True)


@pytest.mark.parametrize(
    "top_k,expected_max_results",
    [
        (1, 1),
        (2, 2),
        (5, 3),  # Only 3 mock results available
        (10, 3),  # Only 3 mock results available
    ],
)
def test_retrieve_top_k_parameter(retriever, mock_vector_store, top_k, expected_max_results):
    """Test that top_k parameter is passed to vector store."""
    result = retriever.retrieve("query", top_k=top_k)

    # Verify top_k was passed to search_similar
    call_args = mock_vector_store.search_similar.call_args
    assert call_args[1]["top_k"] == top_k

    # Verify result doesn't exceed top_k
    assert len(result) <= top_k
    assert len(result) == expected_max_results
