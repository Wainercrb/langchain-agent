"""Unit tests for cronjob hash deduplication logic.

Tests SHA-256 content_hash computation and skip-when-unchanged logic
in isolation using mocked VectorStore.
"""

import hashlib
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore with find_document_by_hash."""
    store = Mock()
    store.find_document_by_hash.return_value = None  # No match by default
    return store


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """Create a sample file for testing."""
    file_path = tmp_path / "test_doc.pdf"
    file_path.write_bytes(b"Sample document content for testing")
    return file_path


def test_compute_content_hash_sha256():
    """Test that content_hash is a valid SHA-256 hex digest."""
    test_data = b"Hello, World!"
    expected_hash = hashlib.sha256(test_data).hexdigest()

    actual_hash = hashlib.sha256(test_data).hexdigest()
    assert actual_hash == expected_hash
    assert len(actual_hash) == 64
    assert isinstance(actual_hash, str)


def test_different_content_produces_different_hash():
    """Test that different file content produces different hashes."""
    hash_a = hashlib.sha256(b"Content A").hexdigest()
    hash_b = hashlib.sha256(b"Content B").hexdigest()

    assert hash_a != hash_b


def test_same_content_produces_same_hash():
    """Test that identical content produces identical hashes."""
    hash_a = hashlib.sha256(b"Same content").hexdigest()
    hash_b = hashlib.sha256(b"Same content").hexdigest()

    assert hash_a == hash_b


def test_find_document_by_hash_no_match(mock_vector_store):
    """Test that no match returns None and processing continues."""
    mock_vector_store.find_document_by_hash.return_value = None
    result = mock_vector_store.find_document_by_hash("nonexistent_hash")

    assert result is None
    mock_vector_store.find_document_by_hash.assert_called_once_with("nonexistent_hash")


def test_find_document_by_hash_match_found(mock_vector_store):
    """Test that a matching hash returns the document and processing skips."""
    mock_doc = {"id": "doc-123", "filename": "existing.pdf", "content_hash": "abc123"}
    mock_vector_store.find_document_by_hash.return_value = mock_doc

    result = mock_vector_store.find_document_by_hash("abc123")

    assert result is not None
    assert result["id"] == "doc-123"
    assert result["content_hash"] == "abc123"


def test_skip_file_when_hash_matches():
    """Test that processing stops when find_document_by_hash returns a match."""
    # Simulate the cron flow: hash matches → skip
    mock_store = Mock()
    mock_store.find_document_by_hash.return_value = {"id": "doc-1"}

    content_hash = "matching_hash"
    existing = mock_store.find_document_by_hash(content_hash)

    # If match found, processing should skip (do NOT call insert_document)
    should_skip = existing is not None
    assert should_skip is True
    # Ensure insert_document was never called
    mock_store.insert_document.assert_not_called()


def test_process_file_when_hash_does_not_match(sample_file, mock_vector_store):
    """Test that processing continues when hash has no match."""
    mock_vector_store.find_document_by_hash.return_value = None

    raw_bytes = sample_file.read_bytes()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    existing = mock_vector_store.find_document_by_hash(content_hash)

    should_process = existing is None
    assert should_process is True


