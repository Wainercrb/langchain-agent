"""Integration tests for health check with LangSmith connectivity."""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from server import app


@pytest.fixture
def client():
    """Create test client for API."""
    return TestClient(app)


class TestHealthCheckWithLangSmith:
    """Tests for health check reporting LangSmith connectivity."""

    def test_health_check_reports_langsmith_connected(self, client):
        """Should report langsmith_connected=True when LangSmith API responds."""
        mock_langsmith_client = MagicMock()
        mock_langsmith_client.list_projects.return_value = [MagicMock(name="test-project")]

        with patch("langsmith.Client", return_value=mock_langsmith_client):
            response = client.get("/v1/health")

            assert response.status_code == 200
            data = response.json()
            assert data["langsmith_connected"] is True

    def test_health_check_reports_langsmith_disconnected_when_api_fails(self, client):
        """Should report langsmith_connected=False when LangSmith API is unreachable."""
        with patch("langsmith.Client") as mock_client:
            mock_client.return_value.list_projects.side_effect = Exception("API unreachable")

            response = client.get("/v1/health")

            assert response.status_code == 200
            data = response.json()
            assert data["langsmith_connected"] is False

    def test_health_check_skips_langsmith_when_disabled(self, client):
        """Should not attempt LangSmith check when tracing is disabled."""
        with patch("api.dependencies.settings") as mock_settings:
            mock_settings.enable_langsmith_tracing = False
            mock_settings.langsmith_api_key = ""

            response = client.get("/v1/health")

            assert response.status_code == 200
            data = response.json()
            assert data["langsmith_connected"] is False
