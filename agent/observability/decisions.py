"""Thread-safe in-memory decision tracker with bounded FIFO eviction and Supabase persistence.

Maintains a deque of DecisionLogEntry instances, automatically evicting
oldest records when the configured maximum size is exceeded.
Records are persisted to Supabase ai_decisions table and restored on startup.
Persistence is batched: writes occur after N records or T seconds, whichever comes first.
"""

import hashlib
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import settings
from loggers import logger
from models.observability.decisions import DecisionLogEntry, DecisionMetricsResponse
from vector_store import VectorStore


@dataclass
class AggregateStats:
    """Aggregate statistics over a set of decision entries."""

    total_decisions: int = 0
    by_agent_type: Dict[str, int] = field(default_factory=dict)
    by_quality: Dict[str, int] = field(default_factory=dict)
    avg_chain_length: float = 0.0
    top_tools: Dict[str, int] = field(default_factory=dict)


class DecisionTracker:
    """Thread-safe bounded store for AI decision records with Supabase persistence.

    Records are indexed by run_id for fast lookup. When the store exceeds
    the configured maximum size, oldest records are evicted (FIFO).
    All records are persisted to Supabase and restored on startup.
    Persistence is batched to reduce API calls under load.

    Args:
        vector_store: VectorStore instance for decision persistence.
        maxlen: Maximum number of records to keep in memory.
        batch_size: Number of records before triggering a save.
        batch_interval: Maximum seconds between saves.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        maxlen: int = 10000,
        batch_size: int = 10,
        batch_interval: float = 30.0,
    ) -> None:
        self._maxlen = maxlen
        self._store: deque[DecisionLogEntry] = deque(maxlen=maxlen)
        self._index: Dict[str, DecisionLogEntry] = {}
        self._lock = threading.Lock()
        self._eviction_count = 0

        # Batching state
        self._batch_size = batch_size
        self._batch_interval = batch_interval
        self._pending_count = 0
        self._last_save_time = time.time()
        self._persisted_ids: set[str] = set()

        # Persistence via vector store
        self._vector_store = vector_store

        self._load()

    def record(self, entry: DecisionLogEntry) -> None:
        """Add or update a decision record.

        If the run_id already exists, the entry is updated in place
        (used for feedback correlation). If the store is at capacity
        and this is a new entry, the oldest record is evicted first.

        Persistence is batched: saves occur after batch_size records
        or batch_interval seconds, whichever comes first.

        Args:
            entry: DecisionLogEntry to record or update.
        """
        with self._lock:
            if entry.run_id in self._index:
                self._update_entry(entry)
                if entry.user_feedback:
                    self._vector_store.update_ai_decision_feedback(
                        entry.run_id, entry.user_feedback
                    )
                return

            self._add_entry(entry)

            self._pending_count += 1
            should_save = (
                self._pending_count >= self._batch_size
                or (time.time() - self._last_save_time) >= self._batch_interval
            )

        if should_save:
            self._save()

    def _update_entry(self, entry: DecisionLogEntry) -> None:
        """Replace an existing entry in the store (feedback correlation)."""
        self._store = deque(
            (entry if e.run_id == entry.run_id else e for e in self._store),
            maxlen=self._maxlen,
        )
        self._index[entry.run_id] = entry
        logger.debug(f"DecisionTracker: updated entry run_id={entry.run_id}")

    def _add_entry(self, entry: DecisionLogEntry) -> None:
        """Add a new entry, evicting the oldest if at capacity."""
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
        """Query decision records with optional filters and pagination."""
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
        """Retrieve a single decision record by run_id."""
        with self._lock:
            return self._index.get(run_id)

    def get_recent(self, limit: int = 50) -> List[DecisionLogEntry]:
        """Return the most recent decision records."""
        with self._lock:
            return list(self._store)[-limit:]

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
        """Compute a deterministic hash of a query string."""
        return hashlib.sha256(query.encode("utf-8")).hexdigest()[:50]

    def _compute_aggregates(self, entries: List[DecisionLogEntry]) -> AggregateStats:
        """Compute aggregate statistics over a list of decision entries."""
        if not entries:
            return AggregateStats()

        by_agent_type: Dict[str, int] = {}
        by_quality: Dict[str, int] = {}
        tool_counts: Dict[str, int] = {}
        total_chain_length = 0

        for entry in entries:
            by_agent_type[entry.agent_type] = by_agent_type.get(entry.agent_type, 0) + 1
            by_quality[entry.decision_quality.value] = by_quality.get(entry.decision_quality.value, 0) + 1
            total_chain_length += entry.chain_length
            for tool_name in entry.tools_used:
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        return AggregateStats(
            total_decisions=len(entries),
            by_agent_type=by_agent_type,
            by_quality=by_quality,
            avg_chain_length=round(total_chain_length / len(entries), 2),
            top_tools=dict(
                sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
        )

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        """Load persisted decisions from Supabase on startup."""
        try:
            rows = self._vector_store.load_ai_decisions(limit=self._maxlen)

            for row in rows:
                entry = DecisionLogEntry.from_db_row(row)
                if len(self._store) >= self._maxlen:
                    evicted = self._store.popleft()
                    self._index.pop(evicted.run_id, None)
                self._store.append(entry)
                self._index[entry.run_id] = entry
                self._persisted_ids.add(entry.run_id)

            logger.info(f"DecisionTracker: loaded {len(self._store)} records from Supabase")
        except Exception as e:
            logger.warning(f"DecisionTracker: failed to load from Supabase: {e}")

    def _save(self) -> None:
        """Persist pending decisions to Supabase."""
        try:
            with self._lock:
                new_entries = [
                    e for e in self._store
                    if e.run_id not in self._persisted_ids
                ]

                if not new_entries:
                    self._pending_count = 0
                    self._last_save_time = time.time()
                    return

                rows = [e.to_db_row() for e in new_entries]

            count = self._vector_store.insert_ai_decisions(rows)

            with self._lock:
                for e in new_entries:
                    self._persisted_ids.add(e.run_id)
                self._pending_count = 0
                self._last_save_time = time.time()

            if count > 0:
                logger.debug(f"DecisionTracker: persisted {count} records to Supabase")
        except Exception as e:
            logger.warning(f"DecisionTracker: failed to save to Supabase: {e}")
