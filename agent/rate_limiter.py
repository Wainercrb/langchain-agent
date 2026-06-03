"""Rate limiter — reusable utility for API quotas.

Used by both infrastructure (embeddings) and API (middleware).
Moved here from api/middleware.py to break circular import:
container → embeddings → RateLimiter → api/handlers → container
"""

import time
from collections import deque

from config.constants import RATE_LIMITER_WINDOW_SECONDS


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
        """Block if we're at the rate limit, then wait until slot is free."""
        while len(self.request_timestamps) >= self.requests_per_minute:
            wait_time = self.get_reset_timestamp() - time.time()
            if wait_time > 0:
                time.sleep(min(wait_time, 1.0))
            self._purge_old_timestamps()
        self.request_timestamps.append(time.time())
