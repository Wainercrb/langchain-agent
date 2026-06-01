"""Integration tests for container wiring with LLM provider failover."""

import pytest

from infrastructure.llm import OpenRouterProvider, ResilientLLMProvider, GoogleProvider, OpenAIProvider


class TestContainerWiring:
    """Tests for container building correct chain with hardcoded provider list."""

    def test_container_uses_resilient_wrapper(self):
        """Should always wrap providers in ResilientLLMProvider."""
        from infrastructure.container import llm

        assert isinstance(llm, ResilientLLMProvider)

    def test_container_has_available_providers(self):
        """Should have at least one provider available."""
        from infrastructure.container import llm

        assert len(llm._providers) >= 1

    def test_provider_names_match_providers(self):
        """Each provider should have a matching name attribute."""
        from infrastructure.container import llm

        for provider in llm._providers:
            assert hasattr(provider, "name")
            assert isinstance(provider.name, str)
            assert len(provider.name) > 0
