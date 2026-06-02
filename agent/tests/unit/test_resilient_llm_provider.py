"""Unit tests for ResilientLLMProvider failover chain."""

import logging
from unittest.mock import MagicMock

import pytest

from infrastructure.llm.base import LLMProvider, LLMResponse, ResilientLLMProvider
from utils.exceptions import (
    AllProvidersExhaustedError,
    TransientLLMError,
    PermanentLLMError,
)


class FakeProvider(LLMProvider):
    """Fake provider for testing failover behavior."""

    name = "fake"

    def __init__(self, name: str = "fake", model: str = "fake-model"):
        super().__init__(model=model, temperature=0.7)
        self.name = name
        self._chat_model = MagicMock()
        self.invoke_calls = 0
        self._side_effect = None
        self._response_content = f"Response from {name}"

    def set_side_effect(self, exception: Exception):
        """Configure the provider to raise an exception on invoke."""
        self._side_effect = exception

    def set_response(self, content: str):
        """Configure the provider to return specific content."""
        self._response_content = content

    @property
    def chat_model(self):
        return self._chat_model

    def invoke(self, messages, **kwargs):
        self.invoke_calls += 1
        if self._side_effect:
            raise self._side_effect
        return LLMResponse(
            content=self._response_content,
            model=self.model,
            provider=self.name,
        )


class TestResilientLLMProviderSuccessOnFirstProvider:
    """Tests for successful invocation on the first provider."""

    def test_first_provider_succeeds(self):
        """Should return response from first provider without failover."""
        provider1 = FakeProvider("google")
        provider1.set_response("Hello from Google")
        provider2 = FakeProvider("openrouter")

        resilient = ResilientLLMProvider(providers=[provider1, provider2])

        result = resilient.invoke([{"role": "user", "content": "Hi"}])

        assert result.content == "Hello from Google"
        assert result.provider == "google"
        assert provider1.invoke_calls == 1
        assert provider2.invoke_calls == 0

    def test_response_metadata_includes_provider(self):
        """Should include provider name in response metadata."""
        provider1 = FakeProvider("openrouter")
        provider2 = FakeProvider("openai")

        resilient = ResilientLLMProvider(providers=[provider1, provider2])

        result = resilient.invoke([{"role": "user", "content": "Hi"}])

        assert result.metadata.get("provider") == "openrouter"
        assert result.metadata.get("failover") is False


class TestResilientLLMProviderFailoverOnTransientError:
    """Tests for failover behavior on transient errors."""

    def test_failover_to_second_provider(self):
        """Should try next provider when first raises TransientLLMError."""
        provider1 = FakeProvider("google")
        provider1.set_side_effect(
            TransientLLMError("Rate limited", provider="google")
        )
        provider2 = FakeProvider("openrouter")
        provider2.set_response("Hello from OpenRouter")

        resilient = ResilientLLMProvider(providers=[provider1, provider2])

        result = resilient.invoke([{"role": "user", "content": "Hi"}])

        assert result.content == "Hello from OpenRouter"
        assert result.provider == "openrouter"
        assert provider1.invoke_calls == 1
        assert provider2.invoke_calls == 1

    def test_failover_metadata_includes_failed_provider(self):
        """Should log which provider failed during failover."""
        provider1 = FakeProvider("google")
        provider1.set_side_effect(
            TransientLLMError("Timeout", provider="google")
        )
        provider2 = FakeProvider("openrouter")
        provider2.set_response("Hello from OpenRouter")

        resilient = ResilientLLMProvider(providers=[provider1, provider2])

        result = resilient.invoke([{"role": "user", "content": "Hi"}])

        assert result.metadata.get("failover") is True
        assert result.metadata.get("failed_providers") == ["google"]


class TestResilientLLMProviderNoFailoverOnPermanentError:
    """Tests for permanent error handling (no failover)."""

    def test_permanent_error_raises_immediately(self):
        """Should raise PermanentLLMError without trying next provider."""
        provider1 = FakeProvider("google")
        provider1.set_side_effect(
            PermanentLLMError("Invalid API key", provider="google")
        )
        provider2 = FakeProvider("openrouter")

        resilient = ResilientLLMProvider(providers=[provider1, provider2])

        with pytest.raises(PermanentLLMError):
            resilient.invoke([{"role": "user", "content": "Hi"}])

        assert provider1.invoke_calls == 1
        assert provider2.invoke_calls == 0


class TestResilientLLMProviderAllExhausted:
    """Tests for all providers exhausted scenario."""

    def test_all_providers_exhausted_raises_aggregated_error(self):
        """Should raise AllProvidersExhaustedError when all providers fail."""
        provider1 = FakeProvider("google")
        provider1.set_side_effect(
            TransientLLMError("Rate limited", provider="google")
        )
        provider2 = FakeProvider("openrouter")
        provider2.set_side_effect(
            TransientLLMError("Timeout", provider="openrouter")
        )

        resilient = ResilientLLMProvider(providers=[provider1, provider2])

        with pytest.raises(AllProvidersExhaustedError) as exc_info:
            resilient.invoke([{"role": "user", "content": "Hi"}])

        assert exc_info.value.attempted_providers == ["google", "openrouter"]
        assert len(exc_info.value.errors) == 2

    def test_all_exhausted_error_contains_individual_errors(self):
        """Should preserve individual errors in aggregated error."""
        provider1 = FakeProvider("google")
        provider1.set_side_effect(
            TransientLLMError("Error 1", provider="google")
        )
        provider2 = FakeProvider("openrouter")
        provider2.set_side_effect(
            TransientLLMError("Error 2", provider="openrouter")
        )

        resilient = ResilientLLMProvider(providers=[provider1, provider2])

        with pytest.raises(AllProvidersExhaustedError) as exc_info:
            resilient.invoke([{"role": "user", "content": "Hi"}])

        errors = exc_info.value.errors
        assert "Error 1" in str(errors[0])
        assert "Error 2" in str(errors[1])


class TestResilientLLMProviderResolveChatModel:
    """Tests for startup chat model resolution."""

    def test_returns_first_working_chat_model(self):
        """Should return chat_model from first available provider."""
        provider1 = FakeProvider("google")
        provider2 = FakeProvider("openrouter")

        resilient = ResilientLLMProvider(providers=[provider1, provider2])

        chat_model = resilient.resolve_chat_model()

        assert chat_model == provider1.chat_model

    def test_skips_unavailable_provider(self):
        """Should skip provider if chat_model raises and try next."""
        provider1 = FakeProvider("google")
        provider1._chat_model = None  # Simulate unavailable
        provider2 = FakeProvider("openrouter")

        resilient = ResilientLLMProvider(providers=[provider1, provider2])

        chat_model = resilient.resolve_chat_model()

        assert chat_model == provider2.chat_model

    def test_raises_when_no_providers_available(self):
        """Should raise clear error when no providers are available."""
        provider1 = FakeProvider("google")
        provider1._chat_model = None
        provider2 = FakeProvider("openrouter")
        provider2._chat_model = None

        resilient = ResilientLLMProvider(providers=[provider1, provider2])

        with pytest.raises(RuntimeError) as exc_info:
            resilient.resolve_chat_model()

        assert "No LLM providers available" in str(exc_info.value)


class TestResilientLLMProviderFailoverLogging:
    """Tests for failover event logging."""

    def test_logs_failover_event(self, caplog):
        """Should log provider failover with both provider names."""
        provider1 = FakeProvider("google")
        provider1.set_side_effect(
            TransientLLMError("Timeout", provider="google")
        )
        provider2 = FakeProvider("openrouter")
        provider2.set_response("Hello from OpenRouter")

        resilient = ResilientLLMProvider(providers=[provider1, provider2])

        with caplog.at_level(logging.INFO):
            resilient.invoke([{"role": "user", "content": "Hi"}])

        assert any("failover" in record.message.lower() for record in caplog.records)
        assert any("google" in record.message.lower() for record in caplog.records)
        assert any("openrouter" in record.message.lower() for record in caplog.records)


class TestResilientLLMProviderNameAttribute:
    """Tests for provider name attribute."""

    def test_provider_name_used_in_response(self):
        """Should use provider.name for response metadata."""
        provider1 = FakeProvider("my-custom-provider")
        provider1.set_response("Hello")

        resilient = ResilientLLMProvider(providers=[provider1])

        result = resilient.invoke([{"role": "user", "content": "Hi"}])

        assert result.metadata.get("provider") == "my-custom-provider"

    def test_provider_name_used_in_failover_log(self):
        """Should use provider.name in failover logging."""
        provider1 = FakeProvider("failing-provider")
        provider1.set_side_effect(
            TransientLLMError("Fail", provider="failing-provider")
        )
        provider2 = FakeProvider("working-provider")
        provider2.set_response("Hello")

        resilient = ResilientLLMProvider(providers=[provider1, provider2])

        result = resilient.invoke([{"role": "user", "content": "Hi"}])

        assert result.metadata.get("failed_providers") == ["failing-provider"]
        assert result.metadata.get("provider") == "working-provider"
