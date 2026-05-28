"""Tests for the exception hierarchy and error classification."""

import pytest

from utils.exceptions import (
    RAGException,
    LLMProviderError,
    TransientLLMError,
    PermanentLLMError,
)


class TestExceptionHierarchy:
    """Verify the unified exception hierarchy under RAGException."""

    def test_llm_provider_error_is_rag_exception(self):
        """LLMProviderError should be a subclass of RAGException."""
        err = LLMProviderError("test", provider="gemini")
        assert isinstance(err, RAGException)
        assert isinstance(err, LLMProviderError)

    def test_transient_error_is_transient(self):
        """TransientLLMError should have is_transient=True."""
        err = TransientLLMError("timeout", provider="openai")
        assert err.is_transient is True
        assert err.provider == "openai"
        assert err.error_code == "LLM_TRANSIENT_ERROR"
        assert isinstance(err, LLMProviderError)
        assert isinstance(err, RAGException)

    def test_permanent_error_is_not_transient(self):
        """PermanentLLMError should have is_transient=False."""
        err = PermanentLLMError("auth failed", provider="gemini")
        assert err.is_transient is False
        assert err.provider == "gemini"
        assert err.error_code == "LLM_PERMANENT_ERROR"
        assert isinstance(err, LLMProviderError)
        assert isinstance(err, RAGException)

    def test_original_error_preserved(self):
        """Original exception should be stored for debugging."""
        original = ConnectionError("network down")
        err = TransientLLMError("connection lost", provider="openrouter", original_error=original)
        assert err.original_error is original
        assert err.original_error.args[0] == "network down"

    def test_provider_stored_in_details(self):
        """Provider name should be stored in details dict."""
        err = LLMProviderError("test error", provider="gemini")
        assert err.details == {"provider": "gemini"}
