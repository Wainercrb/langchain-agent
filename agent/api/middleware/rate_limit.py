"""Rate limiting middleware for FastAPI endpoints."""

import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings
from utils.rate_limiter import RateLimiter

_rate_limiter = RateLimiter(requests_per_minute=settings.rate_limit_requests_per_minute)

_RATE_LIMITED_PATHS = {"/v1/chat", "/v1/rag"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that enforces rate limits on specific paths."""

    async def dispatch(self, request: Request, call_next):
        if not settings.rate_limit_enabled:
            return await call_next(request)

        if request.url.path not in _RATE_LIMITED_PATHS:
            return await call_next(request)

        if not _rate_limiter.is_allowed():
            reset_ts = _rate_limiter.get_reset_timestamp()
            retry_after = max(1, int(reset_ts - time.time()))
            return Response(
                status_code=429,
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(settings.rate_limit_requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(reset_ts)),
                },
                content='{"error": "rate_limit_exceeded", "message": "Too many requests"}',
                media_type="application/json",
            )

        response = await call_next(request)

        remaining = _rate_limiter.get_remaining()
        reset_ts = _rate_limiter.get_reset_timestamp()
        response.headers["X-RateLimit-Limit"] = str(
            settings.rate_limit_requests_per_minute
        )
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(reset_ts))

        return response


rate_limit_middleware = RateLimitMiddleware
