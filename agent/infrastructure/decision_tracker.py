"""Thread-safe in-memory decision tracker with bounded FIFO eviction and disk persistence.

Maintains a deque of DecisionLogEntry instances, automatically evicting
oldest records when the configurable maximum size is exceeded.
Records are persisted to a JSON file and restored on startup.
Persistence is batched: writes occur after N records or T seconds, whichever comes first.
"""

import hashlib
import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config import settings

from models.decision import DecisionLogEntry, DecisionMetricsResponse


class DecisionTracker:
    """Thread-safe bounded store for AI decision records with disk persistence.

    Records are indexed by run_id for fast lookup. When the store exceeds
    the configured maximum size, oldest records are evicted (FIFO).
    All records are persisted to a JSON file and restored on startup.
    Persistence is batched to reduce disk I/O under load.
    """

    def __init__(
        self,
        maxlen: int = 10000,
        persist_path: Optional[str] = None,
        batch_size: int = 10,
        batch_interval: float = 30.0,
    ) -> None:
        self._maxlen = maxlen
        self._store: deque = deque(maxlen=maxlen)
        self._index: Dict[str, DecisionLogEntry] = {}
        self._lock = threading.Lock()
        self._eviction_count = 0
        self._persist_path = Path(persist_path) if persist_path else self._default_path()

        # Batching: save after N records or T seconds, whichever comes first
        self._batch_size = batch_size
        self._batch_interval = batch_interval
        self._pending_count = 0
        self._last_save_time = time.time()

        self._load()

    @staticmethod
    def _default_path() -> Path:
        """Return the default persistence file path."""
        data_dir = Path(os.environ.get("DECISION_TRACKER_DATA_DIR", "data"))
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / "decisions.json"

    def record(self, entry: DecisionLogEntry) -> None:
        """Add a decision record to the store.

        If the run_id already exists, the entry is updated in place (used for
        feedback correlation). If the store is at capacity and this is a new
        entry, the oldest record is evicted first.

        Persistence is batched: saves occur after batch_size records or
        batch_interval seconds, whichever comes first.

        Args:
            entry: DecisionLogEntry to record or update.
        """
        with self._lock:
            if entry.run_id in self._index:
                # Update existing entry (feedback correlation)
                old_entry = self._index[entry.run_id]
                # Replace in deque: rebuild since deque doesn't support random access mutation
                self._store = deque(
                    (entry if e.run_id == entry.run_id else e for e in self._store),
                    maxlen=self._maxlen,
                )
                self._index[entry.run_id] = entry
                logger = __import__("logging").getLogger(__name__)
                logger.debug(f"DecisionTracker: updated entry run_id={entry.run_id}")
                return

            if len(self._store) >= self._maxlen:
                evicted = self._store.popleft()
                self._index.pop(evicted.run_id, None)
                self._eviction_count += 1

            self._store.append(entry)
            self._index[entry.run_id] = entry

            self._pending_count += 1
            should_save = (
                self._pending_count >= self._batch_size
                or (time.time() - self._last_save_time) >= self._batch_interval
            )

        if should_save:
            self._save()

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

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        """Load persisted decisions from disk on startup."""
        if not self._persist_path.exists():
            return

        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            entries = [DecisionLogEntry(**item) for item in data.get("entries", [])]
            self._eviction_count = data.get("eviction_count", 0)

            # Rebuild store and index, respecting maxlen
            for entry in entries:
                if len(self._store) >= self._maxlen:
                    evicted = self._store.popleft()
                    self._index.pop(evicted.run_id, None)
                self._store.append(entry)
                self._index[entry.run_id] = entry

            from infrastructure.logging import logger
            logger.info(f"DecisionTracker: loaded {len(self._store)} records from {self._persist_path}")
        except Exception as e:
            from infrastructure.logging import logger
            logger.warning(f"DecisionTracker: failed to load from {self._persist_path}: {e}")

    def _save(self) -> None:
        """Persist current decisions to disk."""
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "entries": [entry.model_dump() for entry in self._store],
                "eviction_count": self._eviction_count,
            }
            # Write atomically via temp file + rename
            tmp_path = self._persist_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, default=str)
            tmp_path.replace(self._persist_path)

            with self._lock:
                self._pending_count = 0
                self._last_save_time = time.time()
        except Exception as e:
            from infrastructure.logging import logger
            logger.warning(f"DecisionTracker: failed to save to {self._persist_path}: {e}")

    def flush(self) -> None:
        """Force immediate persistence of all pending records.

        Call this on shutdown or before critical operations to ensure
        no records are lost due to batching.
        """
        self._save()
