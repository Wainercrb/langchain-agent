"""Integration tests for rate limit middleware."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware.rate_limit import RateLimitMiddleware, _rate_limiter
from utils.rate_limiter import RateLimiter


@pytest.fixture
def app():
    """Create test app with rate limit middleware."""
    test_app = FastAPI()

    @test_app.get("/v1/chat")
    async def chat():
        return {"status": "ok"}

    @test_app.get("/v1/rag")
    async def rag():
        return {"status": "ok"}

    @test_app.get("/v1/health")
    async def health():
        return {"status": "ok"}

    test_app.add_middleware(RateLimitMiddleware)
    return test_app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestRateLimitMiddleware:
    """Tests for rate limit middleware."""

    def test_requests_within_limit(self, client):
        """Should process requests normally within limit."""
        response = client.get("/v1/chat")

        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    def test_rate_limit_headers_present(self, client):
        """Should include rate limit headers on successful response."""
        response = client.get("/v1/chat")

        limit = int(response.headers["X-RateLimit-Limit"])
        remaining = int(response.headers["X-RateLimit-Remaining"])

        assert limit > 0
        assert remaining >= 0

    def test_unlimited_paths_not_rate_limited(self, client):
        """Should not apply rate limiting to non-targeted paths."""
        response = client.get("/v1/health")

        assert response.status_code == 200
        assert "X-RateLimit-Limit" not in response.headers

    def test_returns_429_after_threshold(self, client):
        """Should return 429 after exceeding rate limit."""
        limiter = _rate_limiter
        original_limit = limiter.requests_per_minute

        try:
            limiter.requests_per_minute = 3
            limiter.request_timestamps.clear()

            for _ in range(3):
                client.get("/v1/chat")

            response = client.get("/v1/chat")

            assert response.status_code == 429
            assert "Retry-After" in response.headers
            assert response.headers["X-RateLimit-Remaining"] == "0"
        finally:
            limiter.requests_per_minute = original_limit
            limiter.request_timestamps.clear()

    def test_429_has_retry_after_header(self, client):
        """Should include Retry-After header on 429 response."""
        limiter = _rate_limiter
        original_limit = limiter.requests_per_minute

        try:
            limiter.requests_per_minute = 2
            limiter.request_timestamps.clear()

            for _ in range(2):
                client.get("/v1/chat")

            response = client.get("/v1/chat")

            assert response.status_code == 429
            retry_after = int(response.headers["Retry-After"])
            assert retry_after > 0
        finally:
            limiter.requests_per_minute = original_limit
            limiter.request_timestamps.clear()
