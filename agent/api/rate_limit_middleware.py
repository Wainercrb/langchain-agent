"""Rate limiting for API quotas — FastAPI middleware.

Combines the RateLimiter utility class and the RateLimitMiddleware
into a single module.
"""

import time
from collections import deque

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings
from config.constants import RATE_LIMITER_WINDOW_SECONDS


class RateLimiter:
    """Rate limiter for API quotas (requests per minute)."""

    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.request_timestamps: deque = deque()  # Track request times (last 60 seconds)
        self.min_interval = (
            RATE_LIMITER_WINDOW_SECONDS / requests_per_minute
        )  # Seconds between requests (~0.6s for 100/min)

    def _purge_old_timestamps(self) -> None:
        """Remove timestamps older than the rate limiter window."""
        cutoff = time.time() - RATE_LIMITER_WINDOW_SECONDS
        while self.request_timestamps and self.request_timestamps[0] < cutoff:
            self.request_timestamps.popleft()

    def is_allowed(self) -> bool:
        """Non-blocking check. Returns True if request should proceed, False if limited."""
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
        oldest = self.request_timestamps[0]
        return oldest + RATE_LIMITER_WINDOW_SECONDS

    def get_remaining(self) -> int:
        """Get remaining requests in current window."""
        self._purge_old_timestamps()
        remaining = self.requests_per_minute - len(self.request_timestamps)
        return max(0, remaining)

    def wait_if_needed(self) -> None:
        """Blocking check — waits if rate limit is reached.

        Used by embedding batch jobs (not API middleware).
        Uses time.sleep() — do NOT call from async request handlers.
        """
        from infrastructure.logging import logger

        current_time = time.time()
        self._purge_old_timestamps()

        # Wait for quota reset if limit reached
        if len(self.request_timestamps) >= self.requests_per_minute:
            oldest_request = self.request_timestamps[0]
            wait_time = (oldest_request + RATE_LIMITER_WINDOW_SECONDS) - current_time
            if wait_time > 0:
                logger.warning(
                    f"Quota limit reached ({len(self.request_timestamps)}/{self.requests_per_minute}). "
                    f"Waiting {wait_time:.1f}s for window to elapse..."
                )
                time.sleep(wait_time)
                current_time = time.time()
                self._purge_old_timestamps()

        # Enforce minimum spacing between requests
        if self.request_timestamps:
            last_request = self.request_timestamps[-1]
            time_since_last = current_time - last_request
            if time_since_last < self.min_interval:
                time.sleep(self.min_interval - time_since_last)
                current_time = time.time()

        self.request_timestamps.append(current_time)


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
