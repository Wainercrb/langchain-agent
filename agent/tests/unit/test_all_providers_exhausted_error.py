"""Unit tests for AllProvidersExhaustedError exception."""

import pytest

from utils.exceptions import (
    AllProvidersExhaustedError,
    LLMProviderError,
    TransientLLMError,
)


class TestAllProvidersExhaustedError:
    """Tests for AllProvidersExhaustedError exception."""

    def test_is_subclass_of_llm_provider_error(self):
        """Should inherit from LLMProviderError."""
        assert issubclass(AllProvidersExhaustedError, LLMProviderError)

    def test_stores_attempted_providers(self):
        """Should store the list of attempted provider names."""
        providers = ["google", "openrouter", "openai"]
        errors = [
            TransientLLMError("timeout", provider="google"),
            TransientLLMError("rate limit", provider="openrouter"),
            TransientLLMError("503", provider="openai"),
        ]

        exc = AllProvidersExhaustedError(
            message="All providers exhausted",
            attempted_providers=providers,
            errors=errors,
        )

        assert exc.attempted_providers == providers

    def test_stores_errors_list(self):
        """Should store the list of errors from each provider."""
        errors = [
            TransientLLMError("timeout", provider="google"),
            TransientLLMError("rate limit", provider="openrouter"),
        ]

        exc = AllProvidersExhaustedError(
            message="All providers exhausted",
            attempted_providers=["google", "openrouter"],
            errors=errors,
        )

        assert exc.errors == errors
        assert len(exc.errors) == 2

    def test_has_correct_error_code(self):
        """Should have ALL_PROVIDERS_EXHAUSTED error code."""
        exc = AllProvidersExhaustedError(
            message="All providers exhausted",
            attempted_providers=["google"],
            errors=[TransientLLMError("timeout", provider="google")],
        )

        assert exc.error_code == "ALL_PROVIDERS_EXHAUSTED"

    def test_provider_attribute_is_all(self):
        """Should set provider to 'all' since all were attempted."""
        exc = AllProvidersExhaustedError(
            message="All providers exhausted",
            attempted_providers=["google", "openrouter"],
            errors=[
                TransientLLMError("timeout", provider="google"),
                TransientLLMError("rate limit", provider="openrouter"),
            ],
        )

        assert exc.provider == "all"

    def test_string_representation(self):
        """Should include error code in string representation."""
        exc = AllProvidersExhaustedError(
            message="All providers exhausted",
            attempted_providers=["google"],
            errors=[TransientLLMError("timeout", provider="google")],
        )

        assert "ALL_PROVIDERS_EXHAUSTED" in str(exc)
        assert "All providers exhausted" in str(exc)
