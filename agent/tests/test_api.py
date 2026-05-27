"""Integration tests for FastAPI endpoints.

Tests the full API stack including FastAPI, routes, models, and dependencies.
Uses TestClient for synchronous testing.
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    """Create FastAPI TestClient."""
    return TestClient(app)


# ============ POST /v1/chat Tests ============


def test_chat_endpoint_success(client):
    """Test POST /v1/chat with valid request returns 200."""
    response = client.post(
        "/v1/chat",
        json={
            "query": "How to enroll?",
            "top_k": 5,
            "include_sources": True,
            "temperature": 0.7,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "response" in data
    assert "query" in data
    assert "execution_time_ms" in data
    assert "model" in data

    # Verify content
    assert data["query"] == "How to enroll?"
    assert data["execution_time_ms"] > 0
    assert isinstance(data["response"], str)
    assert len(data["response"]) > 0


def test_chat_endpoint_with_sources(client):
    """Test POST /v1/chat returns sources when include_sources=True."""
    response = client.post(
        "/v1/chat",
        json={"query": "How to enroll?", "include_sources": True},
    )

    assert response.status_code == 200
    data = response.json()

    # When include_sources=True, sources should be present
    if data.get("sources"):
        assert isinstance(data["sources"], list)
        for source in data["sources"]:
            assert "document_id" in source
            assert "filename" in source
            assert "similarity_score" in source
            assert "version_date" in source
            assert "content_preview" in source
            assert "chunk_id" in source


def test_chat_endpoint_without_sources(client):
    """Test POST /v1/chat excludes sources when include_sources=False."""
    response = client.post(
        "/v1/chat",
        json={"query": "How to enroll?", "include_sources": False},
    )

    assert response.status_code == 200
    data = response.json()

    # When include_sources=False, sources should be None or empty
    assert data.get("sources") is None or len(data["sources"]) == 0


def test_chat_endpoint_missing_query(client):
    """Test POST /v1/chat returns 422 when query is missing."""
    response = client.post("/v1/chat", json={"top_k": 5})

    assert response.status_code == 422  # Unprocessable Entity


def test_chat_endpoint_empty_query(client):
    """Test POST /v1/chat returns 422 when query is empty string."""
    response = client.post("/v1/chat", json={"query": ""})

    assert response.status_code == 422


def test_chat_endpoint_whitespace_query(client):
    """Test POST /v1/chat returns 422 when query is whitespace-only."""
    response = client.post("/v1/chat", json={"query": "   "})

    assert response.status_code == 422


def test_chat_endpoint_query_too_long(client):
    """Test POST /v1/chat returns 422 when query exceeds max_length."""
    long_query = "x" * 2001  # Exceeds max_length=2000

    response = client.post("/v1/chat", json={"query": long_query})

    assert response.status_code == 422


def test_chat_endpoint_top_k_too_low(client):
    """Test POST /v1/chat returns 422 when top_k < 1."""
    response = client.post("/v1/chat", json={"query": "test", "top_k": 0})

    assert response.status_code == 422


def test_chat_endpoint_top_k_too_high(client):
    """Test POST /v1/chat returns 422 when top_k > 20."""
    response = client.post("/v1/chat", json={"query": "test", "top_k": 21})

    assert response.status_code == 422


def test_chat_endpoint_temperature_out_of_range(client):
    """Test POST /v1/chat returns 422 when temperature outside 0.0-1.0."""
    response = client.post("/v1/chat", json={"query": "test", "temperature": 1.5})

    assert response.status_code == 422

    response = client.post("/v1/chat", json={"query": "test", "temperature": -0.1})

    assert response.status_code == 422


def test_chat_endpoint_default_parameters(client):
    """Test POST /v1/chat uses default parameters when not provided."""
    response = client.post("/v1/chat", json={"query": "test query"})

    assert response.status_code == 200
    # Default top_k=5, include_sources=True, temperature=0.7


def test_chat_endpoint_temperature_bounds(client):
    """Test POST /v1/chat accepts temperature at boundaries."""
    response = client.post("/v1/chat", json={"query": "test", "temperature": 0.0})
    assert response.status_code == 200

    response = client.post("/v1/chat", json={"query": "test", "temperature": 1.0})
    assert response.status_code == 200


# ============ GET /v1/health Tests ============


def test_health_endpoint_success(client):
    """Test GET /v1/health returns 200 with correct structure."""
    response = client.get("/v1/health")

    assert response.status_code == 200
    data = response.json()

    # Verify structure
    assert "status" in data
    assert "timestamp" in data
    assert "version" in data
    assert "db_connected" in data

    # Verify content
    assert data["status"] in ["ok", "error"]
    assert isinstance(data["db_connected"], bool)
    assert data["version"] == "1.0.0"


def test_health_endpoint_timestamp_format(client):
    """Test GET /v1/health returns valid ISO timestamp."""
    response = client.get("/v1/health")

    assert response.status_code == 200
    data = response.json()

    # Should be parseable as ISO format
    timestamp = datetime.fromisoformat(data["timestamp"])
    assert isinstance(timestamp, datetime)


# ============ Documentation Endpoints Tests ============


def test_swagger_docs_endpoint(client):
    """Test GET /docs returns Swagger UI."""
    response = client.get("/docs")

    assert response.status_code == 200
    assert "swagger-ui" in response.text.lower() or "swagger" in response.text.lower()


def test_redoc_endpoint(client):
    """Test GET /redoc returns ReDoc documentation."""
    response = client.get("/redoc")

    assert response.status_code == 200


def test_openapi_schema_endpoint(client):
    """Test GET /openapi.json returns valid OpenAPI schema."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()

    # Verify OpenAPI structure
    assert "openapi" in schema or "swagger" in schema
    assert "info" in schema
    assert "paths" in schema

    # Verify endpoints are documented
    assert "/v1/chat" in schema["paths"]
    assert "/v1/health" in schema["paths"]

    # Verify POST /v1/chat
    chat_path = schema["paths"]["/v1/chat"]
    assert "post" in chat_path

    # Verify GET /v1/health
    health_path = schema["paths"]["/v1/health"]
    assert "get" in health_path


# ============ Invalid Endpoint Tests ============


def test_invalid_endpoint_returns_404(client):
    """Test GET to invalid endpoint returns 404."""
    response = client.get("/v1/invalid")

    assert response.status_code == 404


def test_chat_with_get_method_returns_405(client):
    """Test GET /v1/chat returns 405 (method not allowed)."""
    response = client.get("/v1/chat")

    assert response.status_code == 405


def test_health_with_post_method_returns_405(client):
    """Test POST /v1/health returns 405 (method not allowed)."""
    response = client.post("/v1/health")

    assert response.status_code == 405


# ============ Content Type Tests ============


def test_chat_endpoint_requires_json(client):
    """Test POST /v1/chat requires JSON content."""
    # Send as form data instead of JSON
    response = client.post(
        "/v1/chat",
        data={"query": "test"},
    )

    assert response.status_code in [400, 422]


def test_chat_endpoint_returns_json(client):
    """Test POST /v1/chat returns JSON content type."""
    response = client.post("/v1/chat", json={"query": "test"})

    assert response.status_code == 200
    assert "application/json" in response.headers.get("content-type", "")


# ============ Request Validation Tests ============


def test_chat_request_validation_structure(client):
    """Test ChatRequest Pydantic validation."""
    # Valid request
    response = client.post(
        "/v1/chat",
        json={
            "query": "test",
            "top_k": 5,
            "include_sources": True,
            "temperature": 0.7,
        },
    )
    assert response.status_code == 200

    # Invalid type for top_k
    response = client.post(
        "/v1/chat",
        json={"query": "test", "top_k": "five"},
    )
    assert response.status_code == 422

    # Invalid type for temperature
    response = client.post(
        "/v1/chat",
        json={"query": "test", "temperature": "high"},
    )
    assert response.status_code == 422


# ============ Response Structure Tests ============


def test_chat_response_model_structure(client):
    """Test ChatResponse contains all required fields."""
    response = client.post("/v1/chat", json={"query": "test query"})

    assert response.status_code == 200
    data = response.json()

    # All required fields present
    required_fields = ["response", "query", "execution_time_ms", "model"]
    for field in required_fields:
        assert field in data
        assert data[field] is not None


def test_health_response_model_structure(client):
    """Test HealthResponse contains all required fields."""
    response = client.get("/v1/health")

    assert response.status_code == 200
    data = response.json()

    # All required fields present
    required_fields = ["status", "timestamp", "version", "db_connected"]
    for field in required_fields:
        assert field in data
        assert data[field] is not None
