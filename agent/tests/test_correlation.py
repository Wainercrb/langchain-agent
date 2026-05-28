"""Tests for correlation ID utilities."""

import pytest

from utils.correlation import get_correlation_id, set_correlation_id, correlation_id_var


class TestCorrelationId:
    """Verify correlation ID generation and propagation."""

    def setup_method(self):
        """Reset correlation ID before each test."""
        correlation_id_var.set("")

    def test_auto_generates_id_when_empty(self):
        """Should generate an ID if none is set."""
        cid = get_correlation_id()
        assert isinstance(cid, str)
        assert len(cid) > 0

    def test_set_and_get(self):
        """Should return the ID that was set."""
        set_correlation_id("test-123")
        assert get_correlation_id() == "test-123"

    def test_consistent_within_context(self):
        """Should return the same ID on repeated calls."""
        first = get_correlation_id()
        second = get_correlation_id()
        assert first == second

    def test_set_overwrites_auto_generated(self):
        """Setting an ID should override the auto-generated one."""
        auto_id = get_correlation_id()
        set_correlation_id("custom-id")
        assert get_correlation_id() == "custom-id"
        assert get_correlation_id() != auto_id
