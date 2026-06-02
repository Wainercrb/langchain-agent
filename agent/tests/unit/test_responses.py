"""Unit tests for response model serialization."""

import pytest

from models.responses import MetricsResponse


class TestMetricsResponse:
    """Tests for MetricsResponse model with token fields."""

    def test_metrics_response_default_values(self):
        """Should create MetricsResponse with default zero values."""
        response = MetricsResponse()

        assert response.request_count == 0
        assert response.error_count == 0
        assert response.avg_latency_ms == 0.0
        assert response.total_input_tokens == 0
        assert response.total_output_tokens == 0
        assert response.avg_tokens_per_request == 0.0
        assert response.langsmith_audit_url is None

    def test_metrics_response_with_token_values(self):
        """Should serialize token fields with provided values."""
        response = MetricsResponse(
            request_count=10,
            error_count=1,
            avg_latency_ms=150.5,
            total_input_tokens=5000,
            total_output_tokens=3000,
            avg_tokens_per_request=800.0,
        )

        assert response.request_count == 10
        assert response.total_input_tokens == 5000
        assert response.total_output_tokens == 3000
        assert response.avg_tokens_per_request == 800.0

    def test_metrics_response_with_langsmith_audit_url(self):
        """Should include langsmith_audit_url when provided."""
        response = MetricsResponse(
            langsmith_audit_url="https://smith.langchain.com/o/default/projects/p/my-project"
        )

        assert response.langsmith_audit_url == "https://smith.langchain.com/o/default/projects/p/my-project"

    def test_metrics_response_serializes_to_dict(self):
        """Should serialize all fields including token fields to dict."""
        response = MetricsResponse(
            request_count=5,
            total_input_tokens=1000,
            total_output_tokens=500,
            avg_tokens_per_request=300.0,
            langsmith_audit_url="https://example.com",
        )

        data = response.model_dump()
        assert data["total_input_tokens"] == 1000
        assert data["total_output_tokens"] == 500
        assert data["avg_tokens_per_request"] == 300.0
        assert data["langsmith_audit_url"] == "https://example.com"

    def test_metrics_response_serializes_to_json(self):
        """Should serialize all fields including token fields to JSON."""
        response = MetricsResponse(
            request_count=5,
            total_input_tokens=1000,
            total_output_tokens=500,
            avg_tokens_per_request=300.0,
        )

        json_str = response.model_dump_json()
        assert '"total_input_tokens":1000' in json_str
        assert '"total_output_tokens":500' in json_str
        assert '"avg_tokens_per_request":300.0' in json_str
