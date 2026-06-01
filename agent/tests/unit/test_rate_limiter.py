"""Unit tests for rate limiter."""

import pytest
import time
from unittest.mock import patch

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
        
        assert elapsed >= 0.9


class TestIsAllowed:
    """Tests for RateLimiter.is_allowed() non-blocking check."""

    def test_returns_true_within_limit(self):
        """Should return True when under the rate limit."""
        limiter = RateLimiter(requests_per_minute=10)
        assert limiter.is_allowed() is True

    def test_returns_false_after_threshold(self):
        """Should return False after exceeding the rate limit."""
        limiter = RateLimiter(requests_per_minute=5)
        
        for _ in range(5):
            limiter.is_allowed()
        
        assert limiter.is_allowed() is False

    def test_tracks_request_count(self):
        """Should increment request count on each is_allowed call."""
        limiter = RateLimiter(requests_per_minute=10)
        
        limiter.is_allowed()
        limiter.is_allowed()
        limiter.is_allowed()
        
        assert limiter.get_request_count_this_minute() == 3

    def test_resets_after_window(self):
        """Should allow requests again after the window expires."""
        limiter = RateLimiter(requests_per_minute=5)
        
        for _ in range(5):
            limiter.is_allowed()
        
        assert limiter.is_allowed() is False
        
        with patch('utils.rate_limiter.time.time', return_value=time.time() + 61):
            assert limiter.is_allowed() is True

    def test_get_remaining(self):
        """Should return correct remaining count."""
        limiter = RateLimiter(requests_per_minute=10)
        
        assert limiter.get_remaining() == 10
        
        limiter.is_allowed()
        limiter.is_allowed()
        
        assert limiter.get_remaining() == 8

    def test_get_remaining_never_negative(self):
        """Should never return negative remaining."""
        limiter = RateLimiter(requests_per_minute=3)
        
        for _ in range(10):
            limiter.is_allowed()
        
        assert limiter.get_remaining() == 0

    def test_get_reset_timestamp(self):
        """Should return future timestamp for window reset."""
        limiter = RateLimiter(requests_per_minute=10)
        
        reset_ts = limiter.get_reset_timestamp()
        now = time.time()
        
        assert reset_ts > now
        assert reset_ts <= now + 60
