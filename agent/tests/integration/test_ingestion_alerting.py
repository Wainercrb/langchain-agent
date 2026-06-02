"""Integration tests for ingestion alerting."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from domain.ingestion.pipeline import IngestionResult, IngestionStatus

import cronjob


@pytest.fixture
def mock_alert_service():
    """Mock alert service for testing."""
    cronjob.reset_ingestion_alert_cooldown()
    with patch.object(cronjob, 'alert_service') as mock:
        mock.send_alert = AsyncMock()
        yield mock


class TestIngestionAlerting:
    """Tests for ingestion failure alerting."""

    def test_no_alert_on_success(self, mock_alert_service):
        """Should not send alert when all results succeed."""
        results = [
            IngestionResult(filename="doc1.txt", status=IngestionStatus.SUCCESS, chunk_count=5),
            IngestionResult(filename="doc2.txt", status=IngestionStatus.SKIPPED),
        ]

        cronjob._alert_on_failures(results)

        mock_alert_service.send_alert.assert_not_called()

    def test_alert_sent_on_failure(self, mock_alert_service):
        """Should send alert when ingestion failures detected."""
        results = [
            IngestionResult(filename="doc1.txt", status=IngestionStatus.SUCCESS, chunk_count=5),
            IngestionResult(filename="doc2.txt", status=IngestionStatus.FAILED, error="Parse error"),
            IngestionResult(filename="doc3.txt", status=IngestionStatus.FAILED, error="Embedding failed"),
        ]

        cronjob._alert_on_failures(results)

        mock_alert_service.send_alert.assert_called_once()
        call_kwargs = mock_alert_service.send_alert.call_args
        assert call_kwargs.kwargs["metadata"]["failed_count"] == 2

    def test_alert_message_contains_filename(self, mock_alert_service):
        """Should include failed filenames in alert message."""
        results = [
            IngestionResult(filename="report.pdf", status=IngestionStatus.FAILED, error="Parse error"),
        ]

        cronjob._alert_on_failures(results)

        call_kwargs = mock_alert_service.send_alert.call_args
        assert "report.pdf" in call_kwargs.kwargs["message"]

    def test_alert_message_truncated_to_2000_chars(self, mock_alert_service):
        """Should truncate message to avoid Discord limits."""
        results = [
            IngestionResult(filename="doc.txt", status=IngestionStatus.FAILED, error="x" * 3000),
        ]

        cronjob._alert_on_failures(results)

        call_kwargs = mock_alert_service.send_alert.call_args
        assert len(call_kwargs.kwargs["message"]) <= 2000

    def test_alert_limits_to_5_failures(self, mock_alert_service):
        """Should limit alert details to 5 failures to avoid huge messages."""
        results = [
            IngestionResult(filename=f"doc{i}.txt", status=IngestionStatus.FAILED, error=f"Error {i}")
            for i in range(10)
        ]

        cronjob._alert_on_failures(results)

        call_kwargs = mock_alert_service.send_alert.call_args
        message = call_kwargs.kwargs["message"]
        # Should mention 10 total but only list 5
        assert "10 file(s) failed" in message
        assert message.count("- **doc") <= 5

    def test_alert_handles_missing_error(self, mock_alert_service):
        """Should handle failures without error message."""
        results = [
            IngestionResult(filename="doc.txt", status=IngestionStatus.FAILED),
        ]

        cronjob._alert_on_failures(results)

        call_kwargs = mock_alert_service.send_alert.call_args
        assert "Unknown error" in call_kwargs.kwargs["message"]
