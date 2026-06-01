"""Integration tests for health check with LLM failover chain."""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from server import app
from infrastructure.llm import LLMResponse
from utils.exceptions import TransientLLMError, AllProvidersExhaustedError


@pytest.fixture
def client():
    """Create test client for API."""
    return TestClient(app)


class TestHealthCheckWithFailoverChain:
    """Tests for health check reporting chain status."""

    def test_health_check_reports_llm_connected(self, client):
        """Should report llm_connected=True when LLM provider succeeds."""
        mock_response = LLMResponse(
            content="pong",
            model="test-model",
            provider="openrouter",
        )

        with patch("api.dependencies.llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response

            response = client.get("/v1/health")

            assert response.status_code == 200
            data = response.json()
            assert data["llm_connected"] is True

    def test_health_check_reports_llm_disconnected_when_all_fail(self, client):
        """Should report llm_connected=False when all providers fail."""
        with patch("api.dependencies.llm") as mock_llm:
            mock_llm.invoke.side_effect = TransientLLMError(
                "All down", provider="all"
            )

            response = client.get("/v1/health")

            assert response.status_code == 200
            data = response.json()
            assert data["llm_connected"] is False
