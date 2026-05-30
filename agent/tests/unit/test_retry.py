"""Unit tests for retry decorator."""

import pytest
from unittest.mock import MagicMock

from utils.retry import retry_llm
from utils.exceptions import TransientLLMError, PermanentLLMError


class TestRetryLLM:
    """Tests for retry_llm decorator."""

    def test_retry_on_transient_error(self):
        """Should retry on transient errors."""
        call_count = 0
        
        @retry_llm()
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TransientLLMError("Temporary failure", provider="test")
            return "success"
        
        result = failing_function()
        
        assert result == "success"
        assert call_count == 3

    def test_no_retry_on_permanent_error(self):
        """Should not retry on permanent errors."""
        call_count = 0
        
        @retry_llm()
        def failing_function():
            nonlocal call_count
            call_count += 1
            raise PermanentLLMError("Permanent failure", provider="test")
        
        with pytest.raises(PermanentLLMError):
            failing_function()
        
        assert call_count == 1

    def test_no_retry_on_success(self):
        """Should not retry on successful execution."""
        call_count = 0
        
        @retry_llm()
        def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_function()
        
        assert result == "success"
        assert call_count == 1

    def test_retry_exhausted(self):
        """Should raise error after max retries exhausted."""
        call_count = 0
        
        @retry_llm()
        def always_failing():
            nonlocal call_count
            call_count += 1
            raise TransientLLMError("Always fails", provider="test")
        
        with pytest.raises(TransientLLMError):
            always_failing()
        
        # Default retries is 3, so total attempts = 4 (1 initial + 3 retries)
        assert call_count >= 3
