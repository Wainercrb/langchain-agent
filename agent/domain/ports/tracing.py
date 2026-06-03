"""TracingOrchestrator protocol — abstraction for LangSmith tracing.

Domain code depends on this Protocol instead of directly importing
LangSmith infrastructure. The implementation is injected at composition root.
"""

from typing import Any, List, Optional, Protocol

from models import SourceDocument


class TracingOrchestrator(Protocol):
    """Protocol for LangSmith tracing orchestration.

    Provides run ID extraction, trace tags, and source document building.
    """

    def extract_run_id(self) -> str:
        """Extract or generate a run ID for this invocation."""
        ...

    def capture_tracing_tags(
        self,
        model_name: str,
        agent_type: str,
        top_k: int,
        temperature: float,
        decision_metadata: Optional[Any] = None,
        pre_run_id: Optional[str] = None,
    ) -> tuple[str, Optional[List[str]]]:
        """Extract LangSmith run ID and apply dynamic tags."""
        ...

    def build_source_documents(
        self,
        documents: list,
        include_sources: bool,
        content_preview_length: int = 200,
    ) -> Optional[List[SourceDocument]]:
        """Build SourceDocument list from retrieved documents."""
        ...
