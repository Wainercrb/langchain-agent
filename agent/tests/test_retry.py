"""Tests for the shared retry decorator."""

import pytest

from utils.retry import retry_llm
from utils.exceptions import TransientLLMError, PermanentLLMError


class TestRetryDecorator:
    """Verify retry behavior for transient vs permanent errors."""

    def test_retries_on_transient_error(self):
        """Should retry up to max_retries on TransientLLMError."""
        call_count = 0

        @retry_llm(max_retries=3, base_wait=1)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TransientLLMError("temporary failure", provider="test")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count == 3

    def test_no_retry_on_permanent_error(self):
        """Should NOT retry on PermanentLLMError — fail immediately."""
        call_count = 0

        @retry_llm(max_retries=3, base_wait=1)
        def permanent_failure():
            nonlocal call_count
            call_count += 1
            raise PermanentLLMError("auth failed", provider="test")

        with pytest.raises(PermanentLLMError):
            permanent_failure()
        assert call_count == 1

    def test_retries_on_generic_exception(self):
        """Should retry generic (non-LLM) exceptions."""
        call_count = 0

        @retry_llm(max_retries=2, base_wait=1)
        def network_error():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("network down")
            return "recovered"

        result = network_error()
        assert result == "recovered"
        assert call_count == 2

    def test_exhausted_retries_raises(self):
        """Should raise after all retries are exhausted."""
        call_count = 0

        @retry_llm(max_retries=2, base_wait=1)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise TransientLLMError("always fails", provider="test")

        with pytest.raises(TransientLLMError):
            always_fails()
        assert call_count == 2
