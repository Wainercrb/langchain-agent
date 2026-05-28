"""Tests for the Console logger exc_info and correlation ID behavior."""

import json
import pytest
from unittest.mock import patch

from services.logging.console import Console
from utils.correlation import set_correlation_id, correlation_id_var


class TestConsoleLogger:
    """Verify structured JSON logging with exc_info and correlation IDs."""

    def setup_method(self):
        """Reset correlation ID before each test."""
        correlation_id_var.set("")
        self.logger = Console()

    def test_includes_correlation_id(self, capsys):
        """Log entries should include the current correlation ID."""
        set_correlation_id("abc-123")
        self.logger.info("test message")
        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert data["correlation_id"] == "abc-123"
        assert data["message"] == "test message"
        assert data["level"] == "INFO"

    def test_exc_info_captures_stack_trace(self, capsys):
        """exc_info=True should include a stack_trace field."""
        try:
            raise ValueError("test error")
        except ValueError:
            self.logger.error("something failed", exc_info=True)

        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert "stack_trace" in data
        assert "ValueError" in data["stack_trace"]
        assert "test error" in data["stack_trace"]

    def test_no_stack_trace_without_exc_info(self, capsys):
        """Without exc_info, no stack_trace field should appear."""
        self.logger.info("normal log")
        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert "stack_trace" not in data

    def test_extra_kwargs_preserved(self, capsys):
        """Extra kwargs should appear in the 'extra' field."""
        self.logger.info("with extra", user_id="u123", action="login")
        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert data["extra"]["user_id"] == "u123"
        assert data["extra"]["action"] == "login"
