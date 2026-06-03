"""Rate limiting for API quotas — utility, no RAG dependency."""

import time
from collections import deque

from config.constants import RATE_LIMITER_WINDOW_SECONDS


class RateLimiter:
    """Rate limiter for API quotas (requests per minute)."""

    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.request_timestamps = deque()  # Track request times (last 60 seconds)
        self.min_interval = (
            RATE_LIMITER_WINDOW_SECONDS / requests_per_minute
        )  # Seconds between requests (~0.6s for 100/min)

    def wait_if_needed(self):
        """Check quota and wait if necessary to stay within rate limit."""
        from infrastructure.logging import logger  # lazy: avoids circular import

        current_time = time.time()

        # Remove timestamps older than 60 seconds
        while (
            self.request_timestamps and self.request_timestamps[0] < current_time - RATE_LIMITER_WINDOW_SECONDS
        ):
            self.request_timestamps.popleft()

        # Check if we need to wait for quota reset
        if len(self.request_timestamps) >= self.requests_per_minute:
            oldest_request = self.request_timestamps[0]
            wait_time = (oldest_request + RATE_LIMITER_WINDOW_SECONDS) - current_time
            if wait_time > 0:
                logger.warning(
                    f"Quota limit reached ({len(self.request_timestamps)}/{self.requests_per_minute}). "
                    f"Waiting {wait_time:.1f}s for minute to elapse..."
                )
                time.sleep(wait_time)
                # Clear old timestamps after waiting
                current_time = time.time()
                while (
                    self.request_timestamps
                    and self.request_timestamps[0] < current_time - RATE_LIMITER_WINDOW_SECONDS
                ):
                    self.request_timestamps.popleft()

        # Also implement minimum spacing between requests to distribute evenly
        if self.request_timestamps:
            last_request = self.request_timestamps[-1]
            time_since_last = current_time - last_request
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                time.sleep(wait_time)
                current_time = time.time()

        # Record this request
        self.request_timestamps.append(current_time)

    def is_allowed(self) -> bool:
        """Non-blocking check. Returns True if request should proceed, False if limited."""
        current_time = time.time()

        while (
            self.request_timestamps and self.request_timestamps[0] < current_time - RATE_LIMITER_WINDOW_SECONDS
        ):
            self.request_timestamps.popleft()

        if len(self.request_timestamps) >= self.requests_per_minute:
            return False

        self.request_timestamps.append(current_time)
        return True

    def get_request_count_this_minute(self) -> int:
        """Get current request count in the last minute."""
        current_time = time.time()
        while (
            self.request_timestamps and self.request_timestamps[0] < current_time - RATE_LIMITER_WINDOW_SECONDS
        ):
            self.request_timestamps.popleft()
        return len(self.request_timestamps)

    def get_reset_timestamp(self) -> float:
        """Get the timestamp when the current window resets."""
        if not self.request_timestamps:
            return time.time() + RATE_LIMITER_WINDOW_SECONDS
        oldest = self.request_timestamps[0]
        return oldest + RATE_LIMITER_WINDOW_SECONDS

    def get_remaining(self) -> int:
        """Get remaining requests in current window."""
        current_time = time.time()
        while (
            self.request_timestamps and self.request_timestamps[0] < current_time - RATE_LIMITER_WINDOW_SECONDS
        ):
            self.request_timestamps.popleft()
        remaining = self.requests_per_minute - len(self.request_timestamps)
        return max(0, remaining)
