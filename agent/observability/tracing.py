"""LangSmith tracing helpers — extract run IDs, apply tags, and build metadata.

This module handles LangSmith-specific tracing concerns only.
Source document formatting lives in ``retrieval.formatting``.
"""

import uuid
from typing import List, Optional

from langsmith.run_trees import RunTree, _context as run_tree_context
from logging import logger

from models.observability.decisions import DecisionLogEntry


def extract_run_id() -> str:
    """Extract the current LangSmith run ID, falling back to a UUID.

    Returns:
        The LangSmith run ID if a trace is active, otherwise a new UUID.
    """
    current_run = run_tree_context.get_current_run_tree()
    return str(current_run.id) if current_run else str(uuid.uuid4())


def capture_tracing_tags(
    model_name: str,
    agent_type: str,
    top_k: int,
    temperature: float,
    decision_metadata: Optional[DecisionLogEntry] = None,
    pre_run_id: Optional[str] = None,
) -> tuple[str, Optional[List[str]]]:
    """Extract run ID and apply dynamic tags to the active LangSmith trace.

    Args:
        model_name: LLM model identifier.
        agent_type: Agent strategy tag (e.g. "rag-chain", "tool-calling").
        top_k: Number of documents retrieved.
        temperature: LLM temperature setting.
        decision_metadata: Optional DecisionLogEntry for quality/tool tags.
        pre_run_id: Optional pre-computed run_id to reuse.

    Returns:
        Tuple of (run_id, langsmith_tags) — tags is None if no active trace.
    """
    current_run = run_tree_context.get_current_run_tree()
    run_id = pre_run_id or (str(current_run.id) if current_run else str(uuid.uuid4()))

    if not current_run:
        return run_id, None

    tags = _build_tags(model_name, agent_type, top_k, temperature, decision_metadata)
    _apply_metadata(current_run, decision_metadata)
    current_run.add_tags(tags)

    return run_id, tags


def _build_tags(
    model_name: str,
    agent_type: str,
    top_k: int,
    temperature: float,
    decision_metadata: Optional[DecisionLogEntry],
) -> List[str]:
    """Build the list of LangSmith tag strings."""
    tags = [
        f"model:{model_name}",
        f"agent:{agent_type}",
        f"top_k:{top_k}",
        f"temperature:{temperature}",
    ]

    if decision_metadata:
        tags.append(f"decision_quality:{decision_metadata.decision_quality.value}")
        tags.append(f"chain_length:{decision_metadata.chain_length}")
        if decision_metadata.tools_used:
            tags.append(f"tools_used:{','.join(decision_metadata.tools_used)}")

    return tags


def _apply_metadata(run: RunTree, decision_metadata: Optional[DecisionLogEntry]) -> None:
    """Attach decision metadata to the active LangSmith run.

    Silently skips if metadata attachment fails — tracing should never
    break the primary request flow.
    """
    if not decision_metadata:
        return

    try:
        run.add_metadata({
            "agent_type": decision_metadata.agent_type,
            "decision_quality": decision_metadata.decision_quality.value,
            "chain_length": decision_metadata.chain_length,
            "tools_used": decision_metadata.tools_used,
            "reasoning_summary": decision_metadata.reasoning_summary,
            "tool_selection_rationale": decision_metadata.tool_selection_rationale,
            "query_preview": decision_metadata.query_preview,
            "latency_ms": decision_metadata.latency_ms,
            "documents_retrieved": decision_metadata.top_k,
        })

        if decision_metadata.chain_tools:
            run.add_metadata({
                "chain_tools": [
                    {
                        "tool": t.tool_name,
                        "order": t.order,
                        "output_summary": t.output_summary,
                    }
                    for t in decision_metadata.chain_tools
                ],
            })
    except Exception as e:
        logger.debug(f"LangSmith metadata attachment failed: {e}")
