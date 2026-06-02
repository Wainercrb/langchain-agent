"""Thread-safe in-memory decision tracker with bounded FIFO eviction.

Maintains a deque of DecisionLogEntry instances, automatically evicting
oldest records when the configurable maximum size is exceeded.
"""

import hashlib
import threading
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

from config import settings

from models.decision import DecisionLogEntry, DecisionMetricsResponse


class DecisionTracker:
    """Thread-safe bounded in-memory store for AI decision records.

    Records are indexed by run_id for fast lookup. When the store exceeds
    the configured maximum size, oldest records are evicted (FIFO).
    """

    def __init__(self, maxlen: int = 10000) -> None:
        self._maxlen = maxlen
        self._store: deque = deque(maxlen=maxlen)
        self._index: Dict[str, DecisionLogEntry] = {}
        self._lock = threading.Lock()
        self._eviction_count = 0

    def record(self, entry: DecisionLogEntry) -> None:
        """Add a decision record to the store.

        If the store is at capacity, the oldest record is evicted first.

        Args:
            entry: DecisionLogEntry to record.
        """
        with self._lock:
            if len(self._store) >= self._maxlen:
                evicted = self._store.popleft()
                self._index.pop(evicted.run_id, None)
                self._eviction_count += 1

            self._store.append(entry)
            self._index[entry.run_id] = entry

    def query(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        tool: Optional[str] = None,
        quality: Optional[str] = None,
        page: int = 1,
        per_page: int = 20,
    ) -> DecisionMetricsResponse:
        """Query decision records with optional filters and pagination.

        Args:
            from_date: ISO 8601 start date filter (inclusive).
            to_date: ISO 8601 end date filter (inclusive).
            tool: Filter by tool name (matches if tool was used in chain).
            quality: Filter by decision_quality value.
            page: Page number (1-indexed).
            per_page: Number of results per page.

        Returns:
            DecisionMetricsResponse with filtered results and pagination metadata.
        """
        with self._lock:
            filtered = list(self._store)

            if from_date:
                filtered = [e for e in filtered if e.timestamp >= from_date]

            if to_date:
                filtered = [e for e in filtered if e.timestamp <= to_date]

            if tool:
                filtered = [e for e in filtered if tool in e.tools_used]

            if quality:
                filtered = [e for e in filtered if e.decision_quality.value == quality]

            total = len(filtered)
            start = (page - 1) * per_page
            end = start + per_page
            page_items = filtered[start:end]

            aggregates = self._compute_aggregates(filtered)

            return DecisionMetricsResponse(
                total=total,
                page=page,
                per_page=per_page,
                decisions=page_items,
                aggregates=aggregates,
            )

    def get_by_run_id(self, run_id: str) -> Optional[DecisionLogEntry]:
        """Retrieve a single decision record by run_id.

        Args:
            run_id: LangSmith run ID to look up.

        Returns:
            DecisionLogEntry if found, None otherwise.
        """
        with self._lock:
            return self._index.get(run_id)

    @property
    def size(self) -> int:
        """Current number of records in the store."""
        with self._lock:
            return len(self._store)

    @property
    def eviction_count(self) -> int:
        """Total number of records evicted due to capacity limits."""
        with self._lock:
            return self._eviction_count

    @staticmethod
    def compute_query_hash(query: str) -> str:
        """Compute a deterministic hash of a query string.

        Args:
            query: Full query text to hash.

        Returns:
            First 50 characters of the SHA-256 hex digest.
        """
        return hashlib.sha256(query.encode("utf-8")).hexdigest()[:50]

    def _compute_aggregates(self, entries: List[DecisionLogEntry]) -> dict:
        """Compute aggregate statistics over a list of decision entries.

        Args:
            entries: List of DecisionLogEntry to aggregate.

        Returns:
            Dictionary with aggregate statistics.
        """
        if not entries:
            return {
                "total_decisions": 0,
                "by_agent_type": {},
                "by_quality": {},
                "avg_chain_length": 0.0,
                "top_tools": {},
            }

        by_agent_type: Dict[str, int] = {}
        by_quality: Dict[str, int] = {}
        tool_counts: Dict[str, int] = {}
        total_chain_length = 0

        for entry in entries:
            by_agent_type[entry.agent_type] = by_agent_type.get(entry.agent_type, 0) + 1
            by_quality[entry.decision_quality.value] = by_quality.get(entry.decision_quality.value, 0) + 1
            total_chain_length += entry.chain_length
            for tool in entry.tools_used:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

        return {
            "total_decisions": len(entries),
            "by_agent_type": by_agent_type,
            "by_quality": by_quality,
            "avg_chain_length": round(total_chain_length / len(entries), 2),
            "top_tools": dict(sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        }
