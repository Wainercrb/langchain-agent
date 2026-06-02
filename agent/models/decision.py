"""Decision tracking models for AI decision metadata capture.

Pydantic models for recording tool selection, agent routing decisions,
and decision quality metrics.
"""

from enum import Enum
from typing import Dict, List, Optional

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
        max_length=50,
        description="Hash of the full query (first 50 chars)",
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
