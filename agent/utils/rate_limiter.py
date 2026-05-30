"""Rate limiting for API quotas — utility, no RAG dependency."""

import time
from collections import deque


class RateLimiter:
    """Rate limiter for API quotas (requests per minute)."""

    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.request_timestamps = deque()  # Track request times (last 60 seconds)
        self.min_interval = (
            60.0 / requests_per_minute
        )  # Seconds between requests (~0.6s for 100/min)

    def wait_if_needed(self):
        """Check quota and wait if necessary to stay within rate limit."""
        from infrastructure.logging import logger  # lazy: avoids circular import

        current_time = time.time()

        # Remove timestamps older than 60 seconds
        while (
            self.request_timestamps and self.request_timestamps[0] < current_time - 60
        ):
            self.request_timestamps.popleft()

        # Check if we need to wait for quota reset
        if len(self.request_timestamps) >= self.requests_per_minute:
            oldest_request = self.request_timestamps[0]
            wait_time = (oldest_request + 60) - current_time
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
                    and self.request_timestamps[0] < current_time - 60
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

    def get_request_count_this_minute(self) -> int:
        """Get current request count in the last minute."""
        current_time = time.time()
        while (
            self.request_timestamps and self.request_timestamps[0] < current_time - 60
        ):
            self.request_timestamps.popleft()
        return len(self.request_timestamps)
