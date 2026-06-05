"""Decision tracking models for AI decision metadata capture.

Pydantic models for recording tool selection, agent routing decisions,
and decision quality metrics.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DecisionQuality(str, Enum):
    """Quality classification for AI decision outcomes."""

    OPTIMAL = "optimal"
    SUBOPTIMAL = "suboptimal"
    POOR = "poor"


class ToolCallRecord(BaseModel):
    """Record of a single tool invocation within a decision chain."""

    tool_name: str = Field(
        ...,
        description="Name of the tool that was invoked",
    )
    tool_input: dict = Field(
        default_factory=dict,
        description="Input parameters passed to the tool",
    )
    output_summary: Optional[str] = Field(
        default=None,
        description="Brief summary of the tool's output",
    )
    order: int = Field(
        default=0,
        description="Execution order within the chain (0-indexed)",
    )


class DecisionLogEntry(BaseModel):
    """Complete record of a single AI decision event."""

    run_id: str = Field(
        ...,
        description="LangSmith run ID for trace correlation",
    )
    agent_type: str = Field(
        ...,
        description="Agent strategy used: 'tool_calling' or 'rag_chain'",
    )
    query_preview: str = Field(
        ...,
        max_length=200,
        description="First 200 characters of the user query",
    )
    query_hash: str = Field(
        ...,
        max_length=64,
        description="SHA-256 hash of the full query (64 hex chars)",
    )
    tools_used: List[str] = Field(
        default_factory=list,
        description="List of tool names used in execution order",
    )
    chain_length: int = Field(
        default=0,
        description="Number of sequential tool calls made",
    )
    chain_tools: List[ToolCallRecord] = Field(
        default_factory=list,
        description="Detailed tool call chain with inputs and outputs",
    )
    decision_quality: DecisionQuality = Field(
        default=DecisionQuality.SUBOPTIMAL,
        description="Quality classification of the decision",
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp of the decision",
    )
    model_used: str = Field(
        ...,
        description="LLM model identifier",
    )
    top_k: int = Field(
        default=5,
        description="Number of documents retrieved",
    )
    temperature: float = Field(
        default=0.7,
        description="LLM temperature setting",
    )
    latency_ms: float = Field(
        ...,
        description="Total execution latency in milliseconds",
    )
    reasoning_summary: Optional[str] = Field(
        default=None,
        description="Summary of the AI's reasoning for tool selection",
    )
    tool_selection_rationale: Optional[str] = Field(
        default=None,
        description="Raw LLM reasoning text explaining WHY tools were selected",
    )
    user_feedback: Optional[dict] = Field(
        default=None,
        description="User feedback linked to this run (like/dislike)",
    )

    # ── Database row conversion ────────────────────────────────────────

    def to_db_row(self) -> Dict[str, Any]:
        """Convert this entry to a Supabase-compatible row dict."""
        ts = self.timestamp
        if isinstance(ts, datetime):
            ts = ts.isoformat()
        elif not isinstance(ts, str):
            ts = datetime.now(timezone.utc).isoformat()

        quality = self.decision_quality.value if hasattr(self.decision_quality, "value") else self.decision_quality

        return {
            "run_id": self.run_id,
            "agent_type": self.agent_type,
            "query_preview": self.query_preview,
            "query_hash": self.query_hash,
            "tools_used": self.tools_used,
            "chain_length": self.chain_length,
            "chain_tools": [
                ct.model_dump() for ct in self.chain_tools
            ],
            "decision_quality": quality,
            "timestamp": ts,
            "model_used": self.model_used,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "latency_ms": self.latency_ms,
            "reasoning_summary": self.reasoning_summary,
            "tool_selection_rationale": self.tool_selection_rationale,
            "user_feedback": self.user_feedback,
        }

    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> "DecisionLogEntry":
        """Create a DecisionLogEntry from a Supabase row dict."""
        ts = row.get("timestamp")
        if isinstance(ts, datetime):
            ts = ts.isoformat()
        elif ts is None:
            ts = datetime.now(timezone.utc).isoformat()

        return cls(
            run_id=row["run_id"],
            agent_type=row["agent_type"],
            query_preview=row["query_preview"],
            query_hash=row["query_hash"],
            tools_used=row.get("tools_used", []),
            chain_length=row.get("chain_length", 0),
            chain_tools=row.get("chain_tools", []),
            decision_quality=row.get("decision_quality", "suboptimal"),
            timestamp=ts,
            model_used=row["model_used"],
            top_k=row.get("top_k", 5),
            temperature=row.get("temperature", 0.7),
            latency_ms=row["latency_ms"],
            reasoning_summary=row.get("reasoning_summary"),
            tool_selection_rationale=row.get("tool_selection_rationale"),
            user_feedback=row.get("user_feedback"),
        )


class DecisionMetricsResponse(BaseModel):
    """Response model for the GET /v1/decisions endpoint."""

    total: int = Field(
        ...,
        ge=0,
        description="Total number of decisions matching the filter",
    )
    page: int = Field(
        ...,
        ge=1,
        description="Current page number",
    )
    per_page: int = Field(
        ...,
        ge=1,
        description="Number of results per page",
    )
    decisions: List[DecisionLogEntry] = Field(
        default_factory=list,
        description="List of decision log entries for the current page",
    )
    aggregates: Optional[dict] = Field(
        default=None,
        description="Aggregate statistics (count by quality level, etc.)",
    )
