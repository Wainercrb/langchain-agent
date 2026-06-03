"""LLM token-usage counters — input/output tokens and per-request averages.

Decoupled from HTTP request counters: this module only knows about token
consumption. `avg_tokens_per_request` uses an internal record counter so
this class has no dependency on `RequestMetrics`.
"""

import threading


class LLMUsageMetrics:
    """Thread-safe in-memory LLM token counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._record_count = 0

    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage for a completed LLM call.

        Negative values are ignored — they indicate a provider metadata bug
        and must not corrupt the counters.

        Args:
            input_tokens: Prompt/input tokens consumed.
            output_tokens: Completion/output tokens generated.
        """
        if input_tokens < 0 or output_tokens < 0:
            return
        with self._lock:
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._record_count += 1

    def snapshot(self) -> dict:
        """Return a point-in-time snapshot of token usage.

        Returns:
            Dictionary with `total_input_tokens`, `total_output_tokens`, and
            `avg_tokens_per_request` (computed against the internal record count).
        """
        with self._lock:
            total_tokens = self._total_input_tokens + self._total_output_tokens
            avg_tokens = (
                round(total_tokens / self._record_count, 2)
                if self._record_count > 0
                else 0.0
            )
            return {
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "avg_tokens_per_request": avg_tokens,
            }


_llm_usage_metrics = LLMUsageMetrics()


def get_llm_usage_metrics() -> LLMUsageMetrics:
    """Return the global LLMUsageMetrics singleton."""
    return _llm_usage_metrics
