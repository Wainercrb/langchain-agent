"""Supabase-backed repository for AI decision persistence.

Provides durable storage for decision records with batch insert support.
Used by DecisionTracker as the persistence backend instead of file-based JSON.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from logging import logger


class SupabaseDecisionRepository:
    """Repository that persists decision records to the ai_decisions table.

    Args:
        supabase_client: Supabase client instance from the composition root.
    """

    def __init__(self, supabase_client) -> None:
        self.client = supabase_client

    def insert(self, record: Dict[str, Any]) -> bool:
        """Insert a single decision record.

        Args:
            record: Dict matching the ai_decisions table schema.

        Returns:
            True on success, False on failure.
        """
        try:
            self.client.table("ai_decisions").insert(record).execute()
            return True
        except Exception as e:
            logger.warning(f"SupabaseDecisionRepository: insert failed: {e}")
            return False

    def insert_batch(self, records: List[Dict[str, Any]]) -> int:
        """Insert multiple decision records in a single batch.

        Args:
            records: List of dicts matching the ai_decisions table schema.

        Returns:
            Number of records successfully inserted.
        """
        if not records:
            return 0

        try:
            self.client.table("ai_decisions").insert(records).execute()
            return len(records)
        except Exception as e:
            logger.warning(f"SupabaseDecisionRepository: batch insert failed: {e}")
            return 0

    def get_by_run_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single decision record by run_id.

        Args:
            run_id: LangSmith run ID.

        Returns:
            Record dict if found, None otherwise.
        """
        try:
            response = (
                self.client.table("ai_decisions")
                .select("*")
                .eq("run_id", run_id)
                .execute()
            )
            results = response.data or []
            return results[0] if results else None
        except Exception as e:
            logger.warning(f"SupabaseDecisionRepository: lookup failed: {e}")
            return None

    def update_user_feedback(self, run_id: str, feedback: Dict[str, Any]) -> bool:
        """Update the user_feedback column for a given run_id.

        Args:
            run_id: LangSmith run ID.
            feedback: Feedback dict (e.g. {"score": 1, "comment": "great"}).

        Returns:
            True on success, False on failure.
        """
        try:
            self.client.table("ai_decisions").update(
                {"user_feedback": feedback}
            ).eq("run_id", run_id).execute()
            return True
        except Exception as e:
            logger.warning(f"SupabaseDecisionRepository: update failed: {e}")
            return False

    def count(self) -> int:
        """Return total number of decision records in the table.

        Returns:
            Record count, or 0 on error.
        """
        try:
            response = (
                self.client.table("ai_decisions")
                .select("*", count="exact")
                .limit(1)
                .execute()
            )
            return response.count or 0
        except Exception as e:
            logger.warning(f"SupabaseDecisionRepository: count failed: {e}")
            return 0

    def load_recent(self, limit: int = 10000) -> List[Dict[str, Any]]:
        """Load the most recent decision records for in-memory cache rebuild.

        Args:
            limit: Maximum number of records to load.

        Returns:
            List of record dicts ordered by timestamp DESC.
        """
        try:
            response = (
                self.client.table("ai_decisions")
                .select("*")
                .order("timestamp", desc=True)
                .limit(limit)
                .execute()
            )
            results = response.data or []
            # Reverse so oldest is first (matches deque append order)
            results.reverse()
            return results
        except Exception as e:
            logger.warning(f"SupabaseDecisionRepository: load failed: {e}")
            return []

    @staticmethod
    def to_row(entry_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a DecisionLogEntry dict to a database row dict.

        Handles type conversions: ISO timestamps -> datetime, enums -> strings, etc.

        Args:
            entry_dict: Dict from DecisionLogEntry.model_dump().

        Returns:
            Dict suitable for Supabase insert.
        """
        timestamp = entry_dict.get("timestamp", datetime.now(timezone.utc).isoformat())
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except (ValueError, TypeError):
                timestamp = datetime.now(timezone.utc)

        decision_quality = entry_dict.get("decision_quality", "suboptimal")
        if hasattr(decision_quality, "value"):
            decision_quality = decision_quality.value

        return {
            "run_id": entry_dict["run_id"],
            "agent_type": entry_dict["agent_type"],
            "query_preview": entry_dict["query_preview"],
            "query_hash": entry_dict["query_hash"],
            "tools_used": entry_dict.get("tools_used", []),
            "chain_length": entry_dict.get("chain_length", 0),
            "chain_tools": entry_dict.get("chain_tools", []),
            "decision_quality": decision_quality,
            "timestamp": timestamp,
            "model_used": entry_dict["model_used"],
            "top_k": entry_dict.get("top_k", 5),
            "temperature": entry_dict.get("temperature", 0.7),
            "latency_ms": entry_dict["latency_ms"],
            "reasoning_summary": entry_dict.get("reasoning_summary"),
            "tool_selection_rationale": entry_dict.get("tool_selection_rationale"),
            "user_feedback": entry_dict.get("user_feedback"),
        }

    @staticmethod
    def from_row(row: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a database row dict to a DecisionLogEntry-compatible dict.

        Args:
            row: Dict from Supabase query result.

        Returns:
            Dict suitable for DecisionLogEntry(**...).
        """
        timestamp = row.get("timestamp")
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        return {
            "run_id": row["run_id"],
            "agent_type": row["agent_type"],
            "query_preview": row["query_preview"],
            "query_hash": row["query_hash"],
            "tools_used": row.get("tools_used", []),
            "chain_length": row.get("chain_length", 0),
            "chain_tools": row.get("chain_tools", []),
            "decision_quality": row.get("decision_quality", "suboptimal"),
            "timestamp": timestamp,
            "model_used": row["model_used"],
            "top_k": row.get("top_k", 5),
            "temperature": row.get("temperature", 0.7),
            "latency_ms": row["latency_ms"],
            "reasoning_summary": row.get("reasoning_summary"),
            "tool_selection_rationale": row.get("tool_selection_rationale"),
            "user_feedback": row.get("user_feedback"),
        }
