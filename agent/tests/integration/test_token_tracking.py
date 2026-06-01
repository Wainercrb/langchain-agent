"""Integration tests for token tracking in routes."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from server import app
from api.metrics import SimpleMetrics, get_metrics
from api.dependencies import get_agent


@pytest.fixture
def client():
    """Create test client for API."""
    return TestClient(app)


@pytest.fixture
def fresh_metrics():
    """Provide a fresh SimpleMetrics instance for each test."""
    fresh = SimpleMetrics()
    with patch("api.routes.get_metrics", return_value=fresh):
        yield fresh


class TestChatTokenRecording:
    """Tests for token extraction and recording in chat endpoint."""

    def test_chat_records_tokens_on_success(self, client, fresh_metrics):
        """Should record token usage when chat succeeds with usage_metadata."""
        mock_response = MagicMock()
        mock_response.response = "Test answer"
        mock_response.sources = None
        mock_response.model = "test-model"
        mock_response.run_id = None
        mock_response.usage_metadata = {
            "input_tokens": 150,
            "output_tokens": 75,
        }

        mock_processor = MagicMock()
        mock_processor.invoke.return_value = mock_response

        app.dependency_overrides[get_agent] = lambda: mock_processor

        try:
            response = client.post("/v1/chat", json={"query": "Test query"})
            assert response.status_code == 200

            snapshot = fresh_metrics.snapshot()
            assert snapshot["total_input_tokens"] == 150
            assert snapshot["total_output_tokens"] == 75
        finally:
            app.dependency_overrides.clear()

    def test_chat_handles_missing_token_data(self, client, fresh_metrics):
        """Should not error when agent response has no usage_metadata."""
        mock_response = MagicMock()
        mock_response.response = "Test answer"
        mock_response.sources = None
        mock_response.model = "test-model"
        mock_response.run_id = None
        del mock_response.usage_metadata

        mock_processor = MagicMock()
        mock_processor.invoke.return_value = mock_response

        app.dependency_overrides[get_agent] = lambda: mock_processor

        try:
            response = client.post("/v1/chat", json={"query": "Test query"})
            assert response.status_code == 200

            snapshot = fresh_metrics.snapshot()
            assert snapshot["total_input_tokens"] == 0
            assert snapshot["total_output_tokens"] == 0
        finally:
            app.dependency_overrides.clear()

    def test_chat_handles_usage_metadata_exception(self, client, fresh_metrics):
        """Should not error when accessing usage_metadata raises exception."""
        mock_response = MagicMock()
        mock_response.response = "Test answer"
        mock_response.sources = None
        mock_response.model = "test-model"
        mock_response.run_id = None
        type(mock_response).usage_metadata = property(
            lambda self: (_ for _ in ()).throw(AttributeError("no usage"))
        )

        mock_processor = MagicMock()
        mock_processor.invoke.return_value = mock_response

        app.dependency_overrides[get_agent] = lambda: mock_processor

        try:
            response = client.post("/v1/chat", json={"query": "Test query"})
            assert response.status_code == 200

            snapshot = fresh_metrics.snapshot()
            assert snapshot["total_input_tokens"] == 0
            assert snapshot["total_output_tokens"] == 0
        finally:
            app.dependency_overrides.clear()


class TestMetricsEndpointTokenFields:
    """Tests for /v1/metrics returning token fields."""

    @pytest.fixture(autouse=True)
    def reset_metrics(self):
        """Reset global metrics singleton before each test in this class."""
        from api.metrics import _metrics
        with _metrics._lock:
            _metrics._request_count = 0
            _metrics._error_count = 0
            _metrics._total_latency_ms = 0.0
            _metrics._total_input_tokens = 0
            _metrics._total_output_tokens = 0

    def test_metrics_returns_token_fields(self, client):
        """GET /v1/metrics should include token fields in response."""
        response = client.get("/v1/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "total_input_tokens" in data
        assert "total_output_tokens" in data
        assert "avg_tokens_per_request" in data
        assert "langsmith_audit_url" in data

    def test_metrics_token_fields_default_to_zero(self, client):
        """Token fields should default to zero on fresh start."""
        response = client.get("/v1/metrics")
        data = response.json()

        assert data["total_input_tokens"] == 0
        assert data["total_output_tokens"] == 0
        assert data["avg_tokens_per_request"] == 0.0

    def test_metrics_includes_langsmith_audit_url_when_configured(self, client):
        """Should include langsmith_audit_url when tracing is enabled."""
        from config import settings
        original_tracing = settings.enable_langsmith_tracing
        original_key = settings.langsmith_api_key
        original_project = settings.langsmith_project

        try:
            settings.enable_langsmith_tracing = True
            settings.langsmith_api_key = "test-key"
            settings.langsmith_project = "test-project"

            response = client.get("/v1/metrics")
            data = response.json()

            assert data["langsmith_audit_url"] is not None
            assert "test-project" in data["langsmith_audit_url"]
        finally:
            settings.enable_langsmith_tracing = original_tracing
            settings.langsmith_api_key = original_key
            settings.langsmith_project = original_project

    def test_metrics_langsmith_audit_url_absent_when_not_configured(self, client):
        """Should have null langsmith_audit_url when tracing is disabled."""
        from config import settings
        original_tracing = settings.enable_langsmith_tracing
        original_key = settings.langsmith_api_key

        try:
            settings.enable_langsmith_tracing = False
            settings.langsmith_api_key = None

            response = client.get("/v1/metrics")
            data = response.json()

            assert data["langsmith_audit_url"] is None
        finally:
            settings.enable_langsmith_tracing = original_tracing
            settings.langsmith_api_key = original_key
