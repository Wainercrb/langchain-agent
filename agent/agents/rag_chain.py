"""RAGChain — legacy retrieve-then-generate agent.

Retrieves documents, formats them as context, and passes them to an LLM.
Tracing and feedback go through the pluggable ObservabilityProvider.
"""

import hashlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from observability.decorator import trace
from observability.provider import ObservabilityProvider

from config.constants import TRUNCATE_CONTENT_PREVIEW, TRUNCATE_QUERY_LOG, TRUNCATE_QUERY_PREVIEW
from loggers.base import Logger
from models import ChatResponse
from models.observability.decisions import DecisionLogEntry, DecisionQuality
from retrieval.formatting import build_source_documents, format_documents_as_context

from retrieval.retriever import Retriever

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on "
    "provided context. If the context does not contain information "
    "needed to answer the question, say so clearly. Always use the "
    "most recently ingested documents as context."
)


class RAGChain:
    """Retrieve-then-generate chain.

    Retrieves documents via the injected Retriever, formats them as
    context, and passes them to the injected LLM.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: Any,
        decision_tracker: Optional[Any] = None,
        logger: Logger = None,
        observability: Optional[ObservabilityProvider] = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._decision_tracker = decision_tracker
        self._observability = observability
        self.logger = logger
        if self.logger:
            self.logger.info("RAGChain initialized")

    @trace(name="RAGChain.invoke", run_type="chain")
    def invoke(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.7,
        include_sources: bool = True,
        version_filter: Optional[datetime] = None,
        latest_only: bool = True,
    ) -> ChatResponse:
        """Process a query through retrieval + generation.

        Args:
            query: User's natural language question.
            top_k: Number of documents to retrieve.
            temperature: LLM creativity (0.0-1.0).
            include_sources: Whether to include source documents in response.
            version_filter: Optional minimum document version date.
            latest_only: Whether to retrieve only the latest document versions.

        Returns:
            ChatResponse with answer, sources, and metadata.
        """
        start_time = time.time()
        try:
            if self.logger:
                self.logger.info(
                    f"RAGChain.invoke: query={query[:TRUNCATE_QUERY_LOG]}..., top_k={top_k}, "
                    f"temperature={temperature}, include_sources={include_sources}, "
                    f"version_filter={version_filter}, latest_only={latest_only}"
                )

            retrieved = self._retriever.retrieve(
                query=query,
                top_k=top_k,
                version_filter=version_filter,
                latest_only=latest_only,
            )
            if self.logger:
                self.logger.debug(f"Retrieved {len(retrieved)} documents")

            context_str = format_documents_as_context(retrieved)
            if self.logger:
                self.logger.debug(f"Formatted context: {len(context_str)} characters")

            response_text, usage_metadata, llm_latency_ms = self._call_llm(
                context_str, query, temperature
            )

            execution_time_ms = (time.time() - start_time) * 1000

            run_id = self._observability.get_current_run_id()

            decision_metadata = self._extract_decision_metadata(
                run_id, query, execution_time_ms, top_k, temperature, len(retrieved)
            )

            tags = self._build_tags(
                model_name=self._llm.model,
                agent_type="rag-chain",
                top_k=top_k,
                temperature=temperature,
                decision_metadata=decision_metadata,
            )
            self._observability.apply_tags(run_id, tags)
            self._apply_decision_metadata(run_id, decision_metadata)

            sources_list = build_source_documents(
                retrieved, include_sources, TRUNCATE_CONTENT_PREVIEW
            )

            if self._decision_tracker and decision_metadata:
                try:
                    self._decision_tracker.record(decision_metadata)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to record decision: {str(e)}")

            if self.logger:
                self.logger.info(
                    f"RAGChain.invoke complete: query={query[:TRUNCATE_QUERY_LOG]}..., "
                    f"time={execution_time_ms:.0f}ms, llm_time={llm_latency_ms:.0f}ms, "
                    f"sources={len(sources_list or [])}"
                )

            return ChatResponse(
                response=response_text,
                query=query,
                sources=sources_list,
                execution_time_ms=execution_time_ms,
                model=self._llm.model,
                run_id=run_id,
                usage_metadata=usage_metadata,
                llm_latency_ms=llm_latency_ms,
                tracing_tags=tags,
                agent_type=decision_metadata.agent_type if decision_metadata else "rag_chain",
                tools_used=decision_metadata.tools_used if decision_metadata else [],
                chain_length=decision_metadata.chain_length if decision_metadata else 0,
                decision_quality=(
                    decision_metadata.decision_quality.value
                    if decision_metadata
                    else DecisionQuality.SUBOPTIMAL.value
                ),
                reasoning_summary=(
                    decision_metadata.reasoning_summary if decision_metadata else None
                ),
            )

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"RAGChain.invoke failed: query={query[:TRUNCATE_QUERY_LOG]}..., error={str(e)}",
                    exc_info=True,
                )
            raise

    def _call_llm(
        self, context_str: str, query: str, temperature: float
    ) -> tuple[str, Optional[Dict[str, Any]], float]:
        """Call the LLM with formatted context and measure latency.

        Returns:
            Tuple of (response_text, usage_metadata, llm_latency_ms).
        """
        system_prompt = _SYSTEM_PROMPT
        user_prompt = (
            f"Context documents:\n\n{context_str}\n\n"
            f"Question: {query}\n\n"
            f"Please answer the question based on the context above."
        )

        if self.logger:
            self.logger.debug("Constructed prompts for LLM")

        llm_start = time.time()
        llm_response = self._llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        llm_latency_ms = (time.time() - llm_start) * 1000

        response_text = (
            llm_response.content
            if hasattr(llm_response, "content")
            else str(llm_response)
        )
        if self.logger:
            self.logger.debug(f"LLM response received: {len(response_text)} chars")

        usage_metadata = None
        if getattr(llm_response, "usage", None):
            usage_metadata = dict(llm_response.usage)
            if self.logger:
                self.logger.info(
                f"Token usage: prompt={llm_response.usage.get('prompt_tokens', llm_response.usage.get('input_tokens', 'N/A'))}, "
                f"completion={llm_response.usage.get('completion_tokens', llm_response.usage.get('output_tokens', 'N/A'))}, "
                f"total={llm_response.usage.get('total_tokens', 'N/A')}"
            )

        return response_text, usage_metadata, llm_latency_ms

    def _build_tags(
        self,
        model_name: str,
        agent_type: str,
        top_k: int,
        temperature: float,
        decision_metadata: Optional[DecisionLogEntry],
    ) -> List[str]:
        """Build the list of tracing tag strings."""
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

    def _apply_decision_metadata(
        self, run_id: str, decision_metadata: Optional[DecisionLogEntry]
    ) -> None:
        """Attach decision metadata to the active trace."""
        if not decision_metadata or not self._observability:
            return

        metadata = {
            "agent_type": decision_metadata.agent_type,
            "decision_quality": decision_metadata.decision_quality.value,
            "chain_length": decision_metadata.chain_length,
            "tools_used": decision_metadata.tools_used,
            "reasoning_summary": decision_metadata.reasoning_summary,
            "tool_selection_rationale": decision_metadata.tool_selection_rationale,
            "query_preview": decision_metadata.query_preview,
            "latency_ms": decision_metadata.latency_ms,
            "documents_retrieved": decision_metadata.top_k,
        }

        self._observability.apply_metadata(run_id, metadata)

    def _extract_decision_metadata(
        self,
        run_id: str,
        query: str,
        execution_time_ms: float,
        top_k: int,
        temperature: float,
        documents_retrieved: int,
    ) -> DecisionLogEntry:
        """Extract decision metadata for RAGChain invocation.

        RAGChain always retrieves documents, so tool selection is implicit.

        Args:
            run_id: Run ID for this invocation.
            query: Original user query.
            execution_time_ms: Total execution time.
            top_k: Number of documents retrieved.
            temperature: LLM temperature setting.
            documents_retrieved: Actual number of documents retrieved.

        Returns:
            DecisionLogEntry with routing metadata.
        """
        return DecisionLogEntry(
            run_id=run_id,
            agent_type="rag_chain",
            query_preview=query[:TRUNCATE_QUERY_PREVIEW],
            query_hash=hashlib.sha256(query.encode()).hexdigest(),
            tools_used=[],
            chain_length=0,
            chain_tools=[],
            decision_quality=DecisionQuality.OPTIMAL if documents_retrieved > 0 else DecisionQuality.SUBOPTIMAL,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_used=self._llm.model,
            top_k=top_k,
            temperature=temperature,
            latency_ms=execution_time_ms,
            reasoning_summary=f"RAG retrieval: {documents_retrieved} docs with top_k={top_k}",
            tool_selection_rationale=(
                f"Retrieved {documents_retrieved} document(s) via vector search "
                f"(top_k={top_k}); "
                f"{'proceeding with grounded generation' if documents_retrieved > 0 else 'no documents retrieved, falling back to direct LLM answer with no context'}"
            ),
        )
