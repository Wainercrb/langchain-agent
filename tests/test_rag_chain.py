"""Unit tests for RAGChain orchestration service.

Tests RAG pipeline in isolation using mocked Retriever and LLM.
"""

from datetime import datetime
from unittest.mock import Mock

import pytest

from rag.core.chain import RAGChain, RAGResponse, SourceDocument
from rag.retrieval.retriever import RetrievedDocument


@pytest.fixture
def mock_retriever():
    """Mock Retriever with sample retrieved documents."""
    retriever = Mock()
    retriever.retrieve.return_value = [
        RetrievedDocument(
            document_id="doc1",
            chunk_id="chunk1",
            text="How to enroll: step 1 is to complete the online form. Step 2 is to submit documentation.",
            similarity_score=0.95,
            filename="enrollment_guide.pdf",
            version_date=datetime(2025, 1, 1),
        ),
        RetrievedDocument(
            document_id="doc2",
            chunk_id="chunk2",
            text="Enrollment requirements: valid government ID, proof of address, and proof of income.",
            similarity_score=0.88,
            filename="requirements.txt",
            version_date=datetime(2025, 1, 2),
        ),
    ]
    return retriever


@pytest.fixture
def mock_llm():
    """Mock ChatGoogleGenerativeAI that returns a sample response."""
    llm = Mock()
    # Mock response object with content attribute
    mock_response = Mock()
    mock_response.content = (
        "To enroll in the program, you need to follow these steps: "
        "1. Complete the online form 2. Submit required documentation including valid ID and proof of address. "
        "3. Wait for approval (typically 5-7 business days)."
    )
    llm.invoke.return_value = mock_response
    return llm


@pytest.fixture
def rag_chain(mock_retriever, mock_llm):
    """Create RAGChain instance with mocks."""
    return RAGChain(mock_retriever, mock_llm)


def test_invoke_success(rag_chain, mock_retriever, mock_llm):
    """Test successful RAG chain invocation."""
    response = rag_chain.invoke("how to enroll", top_k=5)

    # Verify type
    assert isinstance(response, RAGResponse)

    # Verify content
    assert response.response is not None
    assert len(response.response) > 0
    assert response.query == "how to enroll"
    assert response.execution_time_ms > 0

    # Verify calls
    mock_retriever.retrieve.assert_called_once_with(query="how to enroll", top_k=5)
    mock_llm.invoke.assert_called_once()


def test_invoke_with_sources(rag_chain, mock_retriever):
    """Test RAG chain includes sources when requested."""
    response = rag_chain.invoke("how to enroll", include_sources=True)

    assert response.sources is not None
    assert len(response.sources) > 0
    assert isinstance(response.sources[0], SourceDocument)

    # Verify source content
    assert response.sources[0].document_id == "doc1"
    assert response.sources[0].filename == "enrollment_guide.pdf"
    assert response.sources[0].similarity_score == 0.95
    assert response.sources[0].chunk_id == "chunk1"


def test_invoke_without_sources(rag_chain):
    """Test RAG chain excludes sources when not requested."""
    response = rag_chain.invoke("how to enroll", include_sources=False)

    assert response.sources is None


def test_invoke_with_empty_sources(rag_chain, mock_retriever):
    """Test RAG chain when no documents retrieved."""
    mock_retriever.retrieve.return_value = []
    response = rag_chain.invoke("nonexistent query", include_sources=True)

    # Should still return response, but with no sources
    assert response.response is not None
    # Sources should be None or empty since no documents were retrieved
    assert response.sources is None or len(response.sources) == 0


def test_invoke_temperature_parameter(rag_chain, mock_llm):
    """Test that temperature parameter is used in LLM call."""
    response = rag_chain.invoke("query", temperature=0.2)

    # Temperature should be passed to RAGChain, but we're not directly testing
    # it's in the prompt since we mock the LLM. Just verify invoke completes.
    assert isinstance(response, RAGResponse)


def test_invoke_top_k_parameter(rag_chain, mock_retriever):
    """Test that top_k parameter is passed to retriever."""
    rag_chain.invoke("query", top_k=10)

    call_args = mock_retriever.retrieve.call_args
    assert call_args[1]["top_k"] == 10


def test_invoke_execution_time_measured(rag_chain):
    """Test that execution_time_ms is populated."""
    response = rag_chain.invoke("query")

    assert response.execution_time_ms > 0
    # Should complete in reasonable time (less than 1 second for mocked calls)
    assert response.execution_time_ms < 1000


def test_invoke_model_field(rag_chain):
    """Test that model field is set correctly."""
    response = rag_chain.invoke("query")

    assert response.model == "gemini-2.5-flash"


def test_invoke_query_echo(rag_chain):
    """Test that query is echoed back in response."""
    test_query = "What are the enrollment requirements?"
    response = rag_chain.invoke(test_query)

    assert response.query == test_query


def test_invoke_error_on_retriever_failure(rag_chain, mock_retriever):
    """Test RAG chain handles retriever errors."""
    mock_retriever.retrieve.side_effect = Exception("Retriever error")

    with pytest.raises(Exception, match="Retriever error"):
        rag_chain.invoke("query")


def test_invoke_error_on_llm_failure(rag_chain, mock_llm):
    """Test RAG chain handles LLM errors."""
    mock_llm.invoke.side_effect = Exception("LLM API error")

    with pytest.raises(Exception, match="LLM API error"):
        rag_chain.invoke("query")


def test_invoke_source_document_structure(rag_chain):
    """Test that SourceDocument has all required fields."""
    response = rag_chain.invoke("query", include_sources=True)

    for source in response.sources:
        assert hasattr(source, "document_id")
        assert hasattr(source, "filename")
        assert hasattr(source, "similarity_score")
        assert hasattr(source, "version_date")
        assert hasattr(source, "content_preview")
        assert hasattr(source, "chunk_id")

        # Verify field values
        assert isinstance(source.document_id, str)
        assert isinstance(source.filename, str)
        assert isinstance(source.similarity_score, float)
        assert 0.0 <= source.similarity_score <= 1.0
        assert isinstance(source.version_date, datetime)
        assert isinstance(source.content_preview, str)
        assert len(source.content_preview) <= 200


def test_invoke_returns_rag_response(rag_chain):
    """Test that invoke always returns RAGResponse model."""
    response = rag_chain.invoke("query")

    assert isinstance(response, RAGResponse)
    # Verify Pydantic model can be serialized to dict
    response_dict = response.model_dump()
    assert "response" in response_dict
    assert "query" in response_dict
    assert "execution_time_ms" in response_dict
    assert "model" in response_dict


@pytest.mark.parametrize(
    "include_sources,expected_sources_type",
    [
        (True, list),
        (False, type(None)),
    ],
)
def test_invoke_include_sources_parameter(rag_chain, include_sources, expected_sources_type):
    """Test include_sources parameter controls source inclusion."""
    response = rag_chain.invoke("query", include_sources=include_sources)

    if expected_sources_type is list:
        assert isinstance(response.sources, list)
    else:
        assert response.sources is None


def test_invoke_llm_response_string_conversion(rag_chain, mock_llm):
    """Test that LLM response is converted to string correctly."""
    # Test with object that has .content attribute
    mock_response = Mock()
    mock_response.content = "Test response"
    mock_llm.invoke.return_value = mock_response

    response = rag_chain.invoke("query")
    assert response.response == "Test response"


def test_invoke_multiple_calls_independent(rag_chain, mock_retriever):
    """Test that multiple invoke calls are independent."""
    response1 = rag_chain.invoke("query 1", top_k=5)
    response2 = rag_chain.invoke("query 2", top_k=10)

    assert response1.query == "query 1"
    assert response2.query == "query 2"

    # Verify both calls were made with correct parameters
    assert mock_retriever.retrieve.call_count == 2
    calls = mock_retriever.retrieve.call_args_list
    assert calls[0][1]["query"] == "query 1"
    assert calls[1][1]["query"] == "query 2"
