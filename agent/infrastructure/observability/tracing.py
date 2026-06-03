"""LangSmith tracing helpers — infrastructure concern for observability.

Moved from domain/shared/helpers.py to fix layer boundary violation.
Domain should not contain infrastructure concerns like LangSmith tracing.
"""

import uuid
from typing import Any, List, Optional

from langsmith.run_trees import _context as run_tree_context

from models import SourceDocument
from infrastructure.logging import logger


def extract_run_id() -> str:
    """Extract or generate a run ID for this invocation.

    Returns:
        LangSmith run ID or UUID fallback.
    """
    current_run = run_tree_context.get_current_run_tree()
    return str(current_run.id) if current_run else str(uuid.uuid4())


def capture_tracing_tags(
    model_name: str,
    agent_type: str,
    top_k: int,
    temperature: float,
    decision_metadata: Optional[Any] = None,
    pre_run_id: Optional[str] = None,
) -> tuple[str, Optional[List[str]]]:
    """Extract LangSmith run ID and apply dynamic tags.

    Args:
        model_name: Name of the LLM model used.
        agent_type: Agent type tag (e.g. "rag-chain", "tool-calling").
        top_k: Number of documents retrieved.
        temperature: LLM temperature setting.
        decision_metadata: Optional DecisionLogEntry for decision tags.
        pre_run_id: Optional pre-computed run_id to use.

    Returns:
        Tuple of (run_id, langsmith_tags).
    """
    current_run = run_tree_context.get_current_run_tree()
    run_id = pre_run_id or (str(current_run.id) if current_run else str(uuid.uuid4()))

    if not current_run:
        return run_id, None

    langsmith_tags = [
        f"model:{model_name}",
        f"agent:{agent_type}",
        f"top_k:{top_k}",
        f"temperature:{temperature}",
    ]

    if decision_metadata:
        try:
            langsmith_tags.append(f"decision_quality:{decision_metadata.decision_quality.value}")
            langsmith_tags.append(f"chain_length:{decision_metadata.chain_length}")
            if decision_metadata.tools_used:
                langsmith_tags.append(f"tools_used:{','.join(decision_metadata.tools_used)}")

            current_run.add_metadata({
                "agent_type": decision_metadata.agent_type,
                "decision_quality": decision_metadata.decision_quality.value,
                "chain_length": decision_metadata.chain_length,
                "tools_used": decision_metadata.tools_used,
                "reasoning_summary": decision_metadata.reasoning_summary,
                "tool_selection_rationale": getattr(decision_metadata, "tool_selection_rationale", None),
                "query_preview": decision_metadata.query_preview,
                "latency_ms": decision_metadata.latency_ms,
                "documents_retrieved": decision_metadata.top_k,
            })

            if decision_metadata.chain_tools:
                current_run.add_metadata({
                    "chain_tools": [
                        {
                            "tool": t.tool_name,
                            "order": t.order,
                            "output_summary": t.output_summary,
                        }
                        for t in decision_metadata.chain_tools
                    ],
                })
        except Exception:
            pass

    current_run.add_tags(langsmith_tags)
    return run_id, langsmith_tags


def add_decision_metadata_to_run(
    decision_metadata: Any,
) -> None:
    """Add decision quality metadata to the current LangSmith run.

    Args:
        decision_metadata: DecisionLogEntry with routing and quality metadata.
    """
    current_run = run_tree_context.get_current_run_tree()
    if not current_run or not decision_metadata:
        return

    try:
        current_run.add_metadata({
            "agent_type": decision_metadata.agent_type,
            "decision_quality": decision_metadata.decision_quality.value,
            "chain_length": decision_metadata.chain_length,
            "tools_used": decision_metadata.tools_used,
            "reasoning_summary": decision_metadata.reasoning_summary,
            "tool_selection_rationale": getattr(decision_metadata, "tool_selection_rationale", None),
            "query_preview": decision_metadata.query_preview,
            "latency_ms": decision_metadata.latency_ms,
            "documents_retrieved": decision_metadata.top_k,
        })

        if decision_metadata.chain_tools:
            current_run.add_metadata({
                "chain_tools": [
                    {
                        "tool": t.tool_name,
                        "order": t.order,
                        "output_summary": t.output_summary,
                    }
                    for t in decision_metadata.chain_tools
                ],
            })
    except Exception:
        pass


def build_source_documents(
    documents: list,
    include_sources: bool,
    content_preview_length: int = 200,
) -> Optional[List[SourceDocument]]:
    """Build SourceDocument list from retrieved documents.

    Args:
        documents: List of document objects with document_id, filename,
            similarity_score, version_date, text, and chunk_id attributes.
        include_sources: Whether to include source documents in response.
        content_preview_length: Length of content preview to include.

    Returns:
        List of SourceDocument or None if sources not requested or no documents.
    """
    if not include_sources or not documents:
        return None

    sources_list = [
        SourceDocument(
            document_id=doc.document_id,
            filename=doc.filename,
            similarity_score=doc.similarity_score,
            version_date=doc.version_date,
            content_preview=doc.text[:content_preview_length],
            chunk_id=doc.chunk_id,
        )
        for doc in documents
    ]
    logger.debug(f"Formatted {len(sources_list)} source documents")
    return sources_list
