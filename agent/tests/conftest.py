"""Shared pytest fixtures for all tests."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from typing import List

from models.retrieval import RetrievedDocument
from infrastructure.llm.base import LLMProvider, LLMResponse
from infrastructure.embeddings.base import Embeddings
from infrastructure.vector_store.base import VectorStoreBase


# ── Test Data ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_documents() -> List[RetrievedDocument]:
    """Sample retrieved documents for testing."""
    return [
        RetrievedDocument(
            document_id="doc-1",
            chunk_id="chunk-1",
            filename="test1.txt",
            text="This is the first test document about Python programming.",
            similarity_score=0.95,
        ),
        RetrievedDocument(
            document_id="doc-2",
            chunk_id="chunk-2",
            filename="test2.txt",
            text="This is the second test document about machine learning.",
            similarity_score=0.85,
        ),
        RetrievedDocument(
            document_id="doc-3",
            chunk_id="chunk-3",
            filename="test3.txt",
            text="This is the third test document about data science.",
            similarity_score=0.75,
        ),
    ]


@pytest.fixture
def sample_query() -> str:
    """Sample user query for testing."""
    return "Tell me about Python programming"


# ── Mock Services ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm() -> MagicMock:
    """Mock LLM provider for testing."""
    llm = MagicMock(spec=LLMProvider)
    llm.invoke.return_value = LLMResponse(
        content="This is a test response from the LLM.",
        model="test-model",
        provider="test",
        usage={"prompt_tokens": 10, "completion_tokens": 20},
    )
    llm.chat_model = MagicMock()
    return llm


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Mock embeddings provider for testing."""
    embeddings = MagicMock(spec=Embeddings)
    embeddings.embed_query.return_value = [0.1] * 1536  # 1536-dimensional vector
    embeddings.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]
    return embeddings


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Mock vector store for testing."""
    store = MagicMock(spec=VectorStoreBase)
    store.search_similar = AsyncMock(return_value=[])
    store.insert_document = AsyncMock(return_value="doc-id-123")
    store.health_check = AsyncMock(return_value=True)
    return store


# ── Temporary Files ────────────────────────────────────────────────────────────

@pytest.fixture
def temp_txt_file(tmp_path: Path) -> Path:
    """Create a temporary text file for testing."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("This is a test document.\nIt has multiple lines.\nFor testing purposes.")
    return file_path


@pytest.fixture
def temp_md_file(tmp_path: Path) -> Path:
    """Create a temporary markdown file for testing."""
    file_path = tmp_path / "test.md"
    file_path.write_text("# Test Document\n\nThis is a **markdown** document.\n\n- Item 1\n- Item 2")
    return file_path


@pytest.fixture
def temp_csv_file(tmp_path: Path) -> Path:
    """Create a temporary CSV file for testing."""
    file_path = tmp_path / "test.csv"
    file_path.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,Chicago")
    return file_path


# ── Test Configuration ─────────────────────────────────────────────────────────

@pytest.fixture
def test_settings(tmp_path: Path):
    """Create test settings with temporary directories."""
    from config.settings import Settings
    
    # Create temporary directories
    knowledge_dir = tmp_path / "knowledge" / "raw_docs"
    processed_dir = tmp_path / "knowledge" / "processed"
    failed_dir = tmp_path / "knowledge" / "failed"
    
    knowledge_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    failed_dir.mkdir(parents=True)
    
    # Override settings for testing
    return Settings(
        # Required fields with test values
        google_api_key="test-google-key",
        supabase_url="https://test.supabase.co",
        supabase_key="test-key",
        supabase_direct_url="postgresql://test:test@localhost:5432/test",
        
        # Optional fields with test values
        openrouter_api_key="test-openrouter-key",
        openrouter_model="test-model",
        
        # Paths
        knowledge_dir=knowledge_dir,
        processed_dir=processed_dir,
        failed_dir=failed_dir,
        
        # Processing
        chunk_size=100,
        chunk_overlap=20,
        cron_interval_minutes=1,
    )
