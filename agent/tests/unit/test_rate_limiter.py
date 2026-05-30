"""Unit tests for rate limiter."""

import pytest
import time

from utils.rate_limiter import RateLimiter


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_init_default(self):
        """Should initialize with default rate."""
        limiter = RateLimiter()
        assert limiter.requests_per_minute == 100

    def test_init_custom_rate(self):
        """Should initialize with custom rate."""
        limiter = RateLimiter(requests_per_minute=50)
        assert limiter.requests_per_minute == 50

    def test_get_request_count_initial(self):
        """Should return 0 initially."""
        limiter = RateLimiter(requests_per_minute=10)
        assert limiter.get_request_count_this_minute() == 0

    def test_wait_if_needed_tracks_requests(self):
        """Should track requests after wait_if_needed."""
        limiter = RateLimiter(requests_per_minute=100)
        
        limiter.wait_if_needed()
        assert limiter.get_request_count_this_minute() == 1
        
        limiter.wait_if_needed()
        assert limiter.get_request_count_this_minute() == 2

    def test_min_interval_spacing(self):
        """Should enforce minimum interval between requests."""
        limiter = RateLimiter(requests_per_minute=60)  # 1 request per second
        
        start = time.time()
        limiter.wait_if_needed()
        limiter.wait_if_needed()
        elapsed = time.time() - start
        
        # Should have waited at least ~1 second between requests
        # (min_interval = 60/60 = 1.0 seconds)
        assert elapsed >= 0.9  # Allow small timing variance
