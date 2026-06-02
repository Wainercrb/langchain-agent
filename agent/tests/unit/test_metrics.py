"""Unit tests for LLMUsageMetrics token counters."""

from api.metrics.llm_usage import LLMUsageMetrics


class TestRecordTokens:
    """Tests for LLMUsageMetrics.record_tokens()."""

    def test_record_tokens_increments_input_counter(self):
        """Should increment _total_input_tokens by the given amount."""
        metrics = LLMUsageMetrics()
        metrics.record_tokens(input_tokens=100, output_tokens=50)

        snapshot = metrics.snapshot()
        assert snapshot["total_input_tokens"] == 100

    def test_record_tokens_increments_output_counter(self):
        """Should increment _total_output_tokens by the given amount."""
        metrics = LLMUsageMetrics()
        metrics.record_tokens(input_tokens=100, output_tokens=50)

        snapshot = metrics.snapshot()
        assert snapshot["total_output_tokens"] == 50

    def test_record_tokens_accumulates_across_calls(self):
        """Should accumulate token counts across multiple calls."""
        metrics = LLMUsageMetrics()
        metrics.record_tokens(input_tokens=100, output_tokens=50)
        metrics.record_tokens(input_tokens=200, output_tokens=100)
        metrics.record_tokens(input_tokens=300, output_tokens=150)

        snapshot = metrics.snapshot()
        assert snapshot["total_input_tokens"] == 600
        assert snapshot["total_output_tokens"] == 300

    def test_record_tokens_with_zero_values(self):
        """Should handle zero token values without error."""
        metrics = LLMUsageMetrics()
        metrics.record_tokens(input_tokens=0, output_tokens=0)

        snapshot = metrics.snapshot()
        assert snapshot["total_input_tokens"] == 0
        assert snapshot["total_output_tokens"] == 0

    def test_record_tokens_ignores_negative_values(self):
        """Should silently ignore calls where either value is negative."""
        metrics = LLMUsageMetrics()
        metrics.record_tokens(input_tokens=-10, output_tokens=50)
        metrics.record_tokens(input_tokens=100, output_tokens=-5)

        snapshot = metrics.snapshot()
        assert snapshot["total_input_tokens"] == 0
        assert snapshot["total_output_tokens"] == 0


class TestSnapshotTokenAverages:
    """Tests for LLMUsageMetrics.snapshot() token average calculations."""

    def test_snapshot_includes_token_fields(self):
        """Snapshot should include total_input_tokens, total_output_tokens, avg_tokens_per_request."""
        metrics = LLMUsageMetrics()
        snapshot = metrics.snapshot()

        assert "total_input_tokens" in snapshot
        assert "total_output_tokens" in snapshot
        assert "avg_tokens_per_request" in snapshot

    def test_snapshot_token_fields_default_to_zero(self):
        """Token fields should default to zero when no tokens recorded."""
        metrics = LLMUsageMetrics()
        snapshot = metrics.snapshot()

        assert snapshot["total_input_tokens"] == 0
        assert snapshot["total_output_tokens"] == 0
        assert snapshot["avg_tokens_per_request"] == 0.0

    def test_snapshot_avg_tokens_per_request(self):
        """Should calculate avg_tokens_per_request as (input + output) / record_count."""
        metrics = LLMUsageMetrics()
        metrics.record_tokens(input_tokens=100, output_tokens=50)
        metrics.record_tokens(input_tokens=200, output_tokens=100)
        metrics.record_tokens(input_tokens=300, output_tokens=150)

        snapshot = metrics.snapshot()
        # total = (100+50) + (200+100) + (300+150) = 900, count = 3, avg = 300.0
        assert snapshot["avg_tokens_per_request"] == 300.0

    def test_snapshot_avg_tokens_zero_division(self):
        """Should return 0.0 for avg_tokens_per_request when no records exist."""
        metrics = LLMUsageMetrics()
        snapshot = metrics.snapshot()

        assert snapshot["avg_tokens_per_request"] == 0.0

    def test_snapshot_avg_tokens_with_single_record(self):
        """Should calculate correct average with single record."""
        metrics = LLMUsageMetrics()
        metrics.record_tokens(input_tokens=150, output_tokens=75)

        snapshot = metrics.snapshot()
        # total = 150 + 75 = 225, count = 1, avg = 225.0
        assert snapshot["avg_tokens_per_request"] == 225.0
