"""All API middleware — classes and configuration.

Single module for all middleware concerns:
- RateLimiter: reusable rate-limiting utility (also used by embeddings)
- RateLimitMiddleware: FastAPI middleware for /v1/chat and /v1/rag
- TrafficSheddingMiddleware: returns 503 when system is degraded
- CorrelationIdMiddleware: propagates X-Correlation-ID header
- configure_middleware: wires everything to the FastAPI app
"""

import time
from collections import deque

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings
from config.constants import RATE_LIMITER_WINDOW_SECONDS
from infrastructure.logging import logger
from utils.correlation import set_correlation_id, get_correlation_id


# ---------------------------------------------------------------------------
# RateLimiter — imported from infrastructure to break circular dependency
# ---------------------------------------------------------------------------

from infrastructure.rate_limiter import RateLimiter

# ---------------------------------------------------------------------------

class RateLimiter:
    """Rate limiter for API quotas (requests per minute)."""

    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.request_timestamps: deque = deque()
        self.min_interval = RATE_LIMITER_WINDOW_SECONDS / requests_per_minute

    def _purge_old_timestamps(self) -> None:
        """Remove timestamps older than the rate limiter window."""
        cutoff = time.time() - RATE_LIMITER_WINDOW_SECONDS
        while self.request_timestamps and self.request_timestamps[0] < cutoff:
            self.request_timestamps.popleft()

    def is_allowed(self) -> bool:
        """Non-blocking check. Returns True if request should proceed."""
        self._purge_old_timestamps()
        if len(self.request_timestamps) >= self.requests_per_minute:
            return False
        self.request_timestamps.append(time.time())
        return True

    def get_request_count_this_minute(self) -> int:
        """Get current request count in the last minute."""
        self._purge_old_timestamps()
        return len(self.request_timestamps)

    def get_reset_timestamp(self) -> float:
        """Get the timestamp when the current window resets."""
        if not self.request_timestamps:
            return time.time() + RATE_LIMITER_WINDOW_SECONDS
        return self.request_timestamps[0] + RATE_LIMITER_WINDOW_SECONDS

    def get_remaining(self) -> int:
        """Get remaining requests in current window."""
        self._purge_old_timestamps()
        return max(0, self.requests_per_minute - len(self.request_timestamps))

    def wait_if_needed(self) -> None:
        """Blocking check — waits if rate limit is reached.

        Used by embedding batch jobs (not API middleware).
        Uses time.sleep() — do NOT call from async request handlers.
        """
        current_time = time.time()
        self._purge_old_timestamps()

        if len(self.request_timestamps) >= self.requests_per_minute:
            wait_time = (self.request_timestamps[0] + RATE_LIMITER_WINDOW_SECONDS) - current_time
            if wait_time > 0:
                logger.warning(
                    f"Quota limit reached ({len(self.request_timestamps)}/{self.requests_per_minute}). "
                    f"Waiting {wait_time:.1f}s for window to elapse..."
                )
                time.sleep(wait_time)
                current_time = time.time()
                self._purge_old_timestamps()

        if self.request_timestamps:
            time_since_last = current_time - self.request_timestamps[-1]
            if time_since_last < self.min_interval:
                time.sleep(self.min_interval - time_since_last)

        self.request_timestamps.append(time.time())


# ---------------------------------------------------------------------------
# RateLimitMiddleware — FastAPI middleware
# ---------------------------------------------------------------------------

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
        response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(_rate_limiter.get_remaining())
        response.headers["X-RateLimit-Reset"] = str(int(_rate_limiter.get_reset_timestamp()))
        return response


# ---------------------------------------------------------------------------
# TrafficSheddingMiddleware — ASGI middleware
# ---------------------------------------------------------------------------

_SHED_SAFE_PATHS = {"/v1/health", "/v1/monitoring/status", "/v1/llm/circuits"}


class TrafficSheddingMiddleware:
    """Returns 503 when the system is degraded."""

    def __init__(
        self,
        app,
        shed_on_status: list[str] | None = None,
        retry_after_seconds: int = 60,
        enabled: bool = True,
    ) -> None:
        self.app = app
        self._shed_on_status = shed_on_status or ["error"]
        self._retry_after_seconds = retry_after_seconds
        self._enabled = enabled

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not self._enabled:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in _SHED_SAFE_PATHS:
            await self.app(scope, receive, send)
            return

        from infrastructure.container import _monitoring_scheduler

        if _monitoring_scheduler.overall_status in self._shed_on_status:
            logger.warning(f"Traffic shed: status={_monitoring_scheduler.overall_status}, path={path}")
            response = JSONResponse(
                status_code=503,
                content={
                    "error": "service_unavailable",
                    "message": "Service is temporarily degraded. Please try again later.",
                    "status": _monitoring_scheduler.overall_status,
                },
                headers={"Retry-After": str(self._retry_after_seconds)},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# CorrelationIdMiddleware — FastAPI middleware
# ---------------------------------------------------------------------------

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Propagates X-Correlation-ID header across request/response."""

    async def dispatch(self, request: Request, call_next):
        cid = request.headers.get("X-Correlation-ID", "")
        set_correlation_id(cid)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = get_correlation_id()
        return response


# ---------------------------------------------------------------------------
# Configuration — single entry point
# ---------------------------------------------------------------------------

def configure_middleware(app: FastAPI) -> None:
    """Wire all middleware to the FastAPI application."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(
        TrafficSheddingMiddleware,
        shed_on_status=["error"],
        retry_after_seconds=settings.traffic_shedding_retry_after,
        enabled=settings.traffic_shedding_enabled,
    )
    app.add_middleware(CorrelationIdMiddleware)
