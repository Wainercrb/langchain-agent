"""RAGChain — legacy retrieve-then-generate agent.

Retrieves documents, formats them as context, and passes them to an LLM.
Fully traced via LangSmith @traceable decorator.
"""

import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from langsmith import traceable
from langsmith.run_trees import _context as run_tree_context

from models import ChatResponse, SourceDocument
from utils.formatting import format_documents_as_context

from ..retrieval.retriever import Retriever
from infrastructure.logging import logger

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

    def __init__(self, retriever: Retriever, llm: Any) -> None:
        self._retriever = retriever
        self._llm = llm
        logger.info("RAGChain initialized")

    @traceable(name="RAGChain.invoke", run_type="chain")
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
            logger.info(
                f"RAGChain.invoke: query={query[:50]}..., top_k={top_k}, "
                f"temperature={temperature}, include_sources={include_sources}, "
                f"version_filter={version_filter}, latest_only={latest_only}"
            )

            retrieved = self._retriever.retrieve(
                query=query,
                top_k=top_k,
                version_filter=version_filter,
                latest_only=latest_only,
            )
            logger.debug(f"Retrieved {len(retrieved)} documents")

            context_str = format_documents_as_context(retrieved)
            logger.debug(f"Formatted context: {len(context_str)} characters")

            response_text, usage_metadata, llm_latency_ms = self._call_llm(
                context_str, query, temperature
            )

            execution_time_ms = (time.time() - start_time) * 1000

            run_id, langsmith_tags = self._capture_tracing_metadata(
                top_k, temperature
            )

            sources_list = self._format_sources(
                retrieved, include_sources
            )

            logger.info(
                f"RAGChain.invoke complete: query={query[:50]}..., "
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
                langsmith_tags=langsmith_tags,
            )

        except Exception as e:
            logger.error(
                f"RAGChain.invoke failed: query={query[:50]}..., error={str(e)}",
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

        logger.debug("Constructed prompts for LLM")

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
        logger.debug(f"LLM response received: {len(response_text)} chars")

        # Capture token usage
        usage_metadata = None
        if getattr(llm_response, "usage", None):
            usage_metadata = dict(llm_response.usage)
            logger.info(
                f"Token usage: prompt={llm_response.usage.get('prompt_tokens', llm_response.usage.get('input_tokens', 'N/A'))}, "
                f"completion={llm_response.usage.get('completion_tokens', llm_response.usage.get('output_tokens', 'N/A'))}, "
                f"total={llm_response.usage.get('total_tokens', 'N/A')}"
            )

        return response_text, usage_metadata, llm_latency_ms

    def _capture_tracing_metadata(
        self, top_k: int, temperature: float
    ) -> tuple[str, Optional[List[str]]]:
        """Extract LangSmith run ID and apply dynamic tags.

        Returns:
            Tuple of (run_id, langsmith_tags).
        """
        current_run = run_tree_context.get_current_run_tree()
        run_id = str(current_run.id) if current_run else str(uuid.uuid4())

        if not current_run:
            return run_id, None

        langsmith_tags = [
            f"model:{self._llm.model}",
            "agent:rag-chain",
            f"top_k:{top_k}",
            f"temperature:{temperature}",
        ]
        current_run.add_tags(langsmith_tags)
        return run_id, langsmith_tags

    def _format_sources(
        self, retrieved: list, include_sources: bool
    ) -> Optional[List[SourceDocument]]:
        """Format retrieved documents as SourceDocument list.

        Returns None if sources are not requested or no documents were retrieved.
        """
        if not include_sources or not retrieved:
            return None

        sources_list = [
            SourceDocument(
                document_id=doc.document_id,
                filename=doc.filename,
                similarity_score=doc.similarity_score,
                version_date=doc.version_date,
                content_preview=doc.text[:200],
                chunk_id=doc.chunk_id,
            )
            for doc in retrieved
        ]
        logger.debug(f"Formatted {len(sources_list)} source documents")
        return sources_list
