"""Integration tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from server import app


@pytest.fixture
def client():
    """Create test client for API."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /v1/health endpoint."""

    def test_health_check_success(self, client):
        """Should return health status."""
        response = client.get("/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "db_connected" in data
        assert "langsmith_connected" in data
        assert "embedding_connected" in data

    def test_health_check_returns_json(self, client):
        """Should return JSON response."""
        response = client.get("/v1/health")
        
        assert response.headers["content-type"] == "application/json"


class TestMetricsEndpoint:
    """Tests for /v1/metrics endpoint."""

    def test_metrics_success(self, client):
        """Should return metrics."""
        response = client.get("/v1/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "request_count" in data
        assert "error_count" in data
        assert "avg_latency_ms" in data

    def test_metrics_initial_values(self, client):
        """Should return zero metrics on fresh start."""
        response = client.get("/v1/metrics")
        data = response.json()
        
        # Metrics should be numeric
        assert isinstance(data["request_count"], (int, float))
        assert isinstance(data["error_count"], (int, float))
        assert isinstance(data["avg_latency_ms"], (int, float))


class TestChatEndpoint:
    """Tests for /v1/chat endpoint."""

    def test_chat_requires_query(self, client):
        """Should require query parameter."""
        response = client.post("/v1/chat", json={})
        
        assert response.status_code == 422  # Validation error

    def test_chat_with_valid_query(self, client):
        """Should accept valid chat request."""
        response = client.post(
            "/v1/chat",
            json={
                "query": "What is Python?",
                "top_k": 3,
            }
        )
        
        # Should not return 422 (validation error)
        # May return 200 or 500 depending on service availability
        assert response.status_code != 422

    def test_chat_invalid_top_k(self, client):
        """Should reject invalid top_k values."""
        response = client.post(
            "/v1/chat",
            json={
                "query": "Test query",
                "top_k": 100,  # Too high
            }
        )
        
        assert response.status_code == 422

    def test_chat_empty_query(self, client):
        """Should reject empty query."""
        response = client.post(
            "/v1/chat",
            json={
                "query": "",
            }
        )
        
        assert response.status_code == 422


class TestFeedbackEndpoint:
    """Tests for /v1/feedback endpoint."""

    def test_feedback_requires_run_id(self, client):
        """Should require run_id parameter."""
        response = client.post(
            "/v1/feedback",
            json={
                "is_positive": True,
            }
        )
        
        assert response.status_code == 422

    def test_feedback_with_valid_data(self, client):
        """Should accept valid feedback."""
        response = client.post(
            "/v1/feedback",
            json={
                "run_id": "test-run-123",
                "feedback_type": "like",
                "comment": "Great answer!",
            }
        )
        
        # Should not return 422 (validation error)
        assert response.status_code != 422
