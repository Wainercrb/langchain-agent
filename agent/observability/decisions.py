"""Thread-safe in-memory decision store with bounded FIFO eviction and Supabase persistence.

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

from loggers import logger
from models.observability.decisions import DecisionLogEntry, DecisionMetricsResponse
from vector_store import VectorStore


# ── Pure domain functions ─────────────────────────────────────────────


def compute_query_hash(query: str) -> str:
    """Compute a deterministic SHA-256 hash of a query string."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:50]


@dataclass
class _AggregateStats:
    """Aggregate statistics over a set of decision entries (internal)."""

    total_decisions: int = 0
    by_agent_type: Dict[str, int] = field(default_factory=dict)
    by_quality: Dict[str, int] = field(default_factory=dict)
    avg_chain_length: float = 0.0
    top_tools: Dict[str, int] = field(default_factory=dict)


def compute_aggregates(entries: List[DecisionLogEntry]) -> _AggregateStats:
    """Aggregate statistics over a list of decision entries.

    Args:
        entries: Decision entries to analyze.

    Returns:
        _AggregateStats with counts, averages, and top tools.
    """
    if not entries:
        return _AggregateStats()

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

    return _AggregateStats(
        total_decisions=len(entries),
        by_agent_type=by_agent_type,
        by_quality=by_quality,
        avg_chain_length=round(total_chain_length / len(entries), 2),
        top_tools=dict(
            sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ),
    )


# ── Internal store (bounded deque indexed by run_id) ──────────────────


class _DecisionStore:
    """Bounded FIFO store for DecisionLogEntry records, indexed by run_id.

    Caller MUST hold an external lock when calling any method — this class
    is NOT thread-safe by itself. Thread safety is handled by DecisionTracker.

    Args:
        maxlen: Maximum number of records before FIFO eviction kicks in.
    """

    def __init__(self, maxlen: int = 10000) -> None:
        self._maxlen = maxlen
        self._deque: deque[DecisionLogEntry] = deque(maxlen=maxlen)
        self._index: Dict[str, DecisionLogEntry] = {}
        self._eviction_count = 0

    # ── Mutations ────────────────────────────────────────────────────

    def add(self, entry: DecisionLogEntry) -> None:
        """Insert a new entry, evicting the oldest if at capacity."""
        if len(self._deque) >= self._maxlen:
            evicted = self._deque.popleft()
            self._index.pop(evicted.run_id, None)
            self._eviction_count += 1
        self._deque.append(entry)
        self._index[entry.run_id] = entry

    def update(self, entry: DecisionLogEntry) -> None:
        """Replace an existing entry in place (used for feedback correlation)."""
        self._deque = deque(
            (entry if e.run_id == entry.run_id else e for e in self._deque),
            maxlen=self._maxlen,
        )
        self._index[entry.run_id] = entry
        logger.debug("DecisionStore: updated entry run_id=%s", entry.run_id)

    def rebuild_from(self, entries: List[DecisionLogEntry]) -> None:
        """Replace all contents, evicting extras to fit maxlen."""
        self._deque.clear()
        self._index.clear()
        self._eviction_count = 0
        for entry in entries:
            if len(self._deque) >= self._maxlen:
                self._deque.popleft()
            self._deque.append(entry)
            self._index[entry.run_id] = entry

    # ── Queries ──────────────────────────────────────────────────────

    def get_all(self) -> List[DecisionLogEntry]:
        """Return a shallow copy of all entries."""
        return list(self._deque)

    def get_by_run_id(self, run_id: str) -> Optional[DecisionLogEntry]:
        """Lookup an entry by run_id."""
        return self._index.get(run_id)

    def get_recent(self, limit: int = 50) -> List[DecisionLogEntry]:
        """Return the most recent ``limit`` entries."""
        return list(self._deque)[-limit:]

    def contains(self, run_id: str) -> bool:
        """Check if a run_id is already indexed."""
        return run_id in self._index

    # ── Properties ───────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._deque)

    @property
    def eviction_count(self) -> int:
        return self._eviction_count


# ── Internal persistence (Supabase batch save/load) ──────────────────


class _DecisionPersistence:
    """Batched Supabase persistence for decision records.

    Saves happen when either ``batch_size`` new records accumulate or
    ``batch_interval`` seconds elapse — whichever comes first.

    Caller MUST hold an external lock for :meth:`mark_and_check` — saves
    are designed to run outside the lock to avoid I/O under contention.

    Args:
        vector_store: VectorStore instance for DB operations.
        batch_size: Records accumulated before a save triggers.
        batch_interval: Max seconds between saves.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        batch_size: int = 10,
        batch_interval: float = 30.0,
    ) -> None:
        self._vector_store = vector_store
        self._batch_size = batch_size
        self._batch_interval = batch_interval
        self._pending_count = 0
        self._last_save_time = time.time()
        self._persisted_ids: set[str] = set()

    # ── Load ─────────────────────────────────────────────────────────

    def load(self, maxlen: int) -> List[DecisionLogEntry]:
        """Load persisted decisions from Supabase.

        Args:
            maxlen: Max records to load.

        Returns:
            List of DecisionLogEntry loaded from DB, or empty list on failure.
        """
        try:
            rows = self._vector_store.load_ai_decisions(limit=maxlen)
            entries: List[DecisionLogEntry] = []
            for row in rows:
                entry = DecisionLogEntry.from_db_row(row)
                entries.append(entry)
                self._persisted_ids.add(entry.run_id)
            if entries:
                logger.info("DecisionPersistence: loaded %d records from Supabase", len(entries))
            return entries
        except Exception as e:
            logger.warning("DecisionPersistence: failed to load from Supabase: %s", e)
            return []

    # ── Save (batch tracking) ────────────────────────────────────────

    def mark_and_check(self) -> bool:
        """Increment the pending counter and check if a save is due.

        Call inside the lock. The caller then runs :meth:`save` outside the lock.

        Returns:
            True if :meth:`save` should be called.
        """
        self._pending_count += 1
        return (
            self._pending_count >= self._batch_size
            or (time.time() - self._last_save_time) >= self._batch_interval
        )

    def save(self, entries: List[DecisionLogEntry]) -> None:
        """Persist unpended entries to Supabase.

        Only entries not yet marked as persisted are saved.

        Args:
            entries: Full snapshot of all entries — only unpersisted ones are written.
        """
        try:
            new_entries = [e for e in entries if e.run_id not in self._persisted_ids]
            if not new_entries:
                self._reset_batch()
                return

            rows = [e.to_db_row() for e in new_entries]
            count = self._vector_store.insert_ai_decisions(rows)

            for e in new_entries:
                self._persisted_ids.add(e.run_id)
            self._reset_batch()

            if count > 0:
                logger.debug("DecisionPersistence: persisted %d records to Supabase", count)
        except Exception as e:
            logger.warning("DecisionPersistence: failed to save to Supabase: %s", e)

    def persist_feedback(self, run_id: str, feedback: dict) -> None:
        """Directly update feedback for an already-persisted entry."""
        try:
            self._vector_store.update_ai_decision_feedback(run_id, feedback)
        except Exception as e:
            logger.warning("DecisionPersistence: failed to persist feedback: %s", e)

    def _reset_batch(self) -> None:
        self._pending_count = 0
        self._last_save_time = time.time()


# ── Public facade ────────────────────────────────────────────────────


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
        self._store = _DecisionStore(maxlen=maxlen)
        self._persistence = _DecisionPersistence(
            vector_store=vector_store,
            batch_size=batch_size,
            batch_interval=batch_interval,
        )
        self._lock = threading.Lock()

        self._load_initial()

    # ── Public API ───────────────────────────────────────────────────

    def record(self, entry: DecisionLogEntry) -> None:
        """Add or update a decision record.

        If the run_id already exists, the entry is updated in place
        (used for feedback correlation). If the store is at capacity
        and this is a new entry, the oldest record is evicted first.

        Persistence is batched: the actual save fires after accumulating
        ``batch_size`` new records or ``batch_interval`` seconds.

        Args:
            entry: DecisionLogEntry to record or update.
        """
        with self._lock:
            if self._store.contains(entry.run_id):
                self._store.update(entry)
                if entry.user_feedback:
                    self._persistence.persist_feedback(entry.run_id, entry.user_feedback)
                return

            self._store.add(entry)
            should_save = self._persistence.mark_and_check()

            if should_save:
                snapshot = self._store.get_all()

        if should_save:
            self._persistence.save(snapshot)

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
            from_date: Filter by timestamp >= this value (ISO 8601).
            to_date: Filter by timestamp <= this value (ISO 8601).
            tool: Filter by tool name present in tools_used.
            quality: Filter by decision_quality value.
            page: Page number (1-indexed).
            per_page: Results per page.

        Returns:
            DecisionMetricsResponse with filtered, paginated results and aggregates.
        """
        with self._lock:
            filtered = self._store.get_all()

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

        return DecisionMetricsResponse(
            total=total,
            page=page,
            per_page=per_page,
            decisions=page_items,
            aggregates=compute_aggregates(filtered),
        )

    def get_by_run_id(self, run_id: str) -> Optional[DecisionLogEntry]:
        """Retrieve a single decision record by run_id."""
        with self._lock:
            return self._store.get_by_run_id(run_id)

    def get_recent(self, limit: int = 50) -> List[DecisionLogEntry]:
        """Return the most recent decision records."""
        with self._lock:
            return self._store.get_recent(limit)

    @property
    def size(self) -> int:
        """Current number of records in the store."""
        with self._lock:
            return self._store.size

    @property
    def eviction_count(self) -> int:
        """Total number of records evicted due to capacity limits."""
        with self._lock:
            return self._store.eviction_count

    # ── Internal ─────────────────────────────────────────────────────

    def _load_initial(self) -> None:
        """Load persisted decisions from Supabase on startup.

        Called once during __init__ — no concurrent access is possible yet,
        so the lock is not needed.
        """
        maxlen = self._store._maxlen
        entries = self._persistence.load(maxlen)
        if entries:
            self._store.rebuild_from(entries)
