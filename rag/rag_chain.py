"""RAG (Retrieval-Augmented Generation) chain orchestration.

Combines document retrieval with LLM generation to provide contextual answers.
"""

import time
import logging
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

from .retriever import Retriever, RetrievedDocument
from api.models import SourceDocument

logger = logging.getLogger(__name__)


class RAGResponse(BaseModel):
    """Response from RAG chain invocation."""

    response: str = Field(..., description="LLM-generated answer")
    query: str = Field(..., description="Echo of user query")
    sources: Optional[List[SourceDocument]] = Field(
        default=None, description="Retrieved documents used for context"
    )
    execution_time_ms: float = Field(..., description="Total query time in milliseconds")
    model: str = Field(
        default="gemini-2.5-flash", description="LLM model used to generate response"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "response": "To enroll in the program, you need to follow these steps...",
                "query": "How do I enroll?",
                "sources": [
                    {
                        "document_id": "550e8400-e29b-41d4-a716-446655440000",
                        "filename": "enrollment_guide.pdf",
                        "similarity_score": 0.92,
                        "version_date": "2025-01-15T10:30:00",
                        "content_preview": "Enrollment Guidelines: To enroll in our program...",
                        "chunk_id": "550e8400-e29b-41d4-a716-446655440001",
                    }
                ],
                "execution_time_ms": 2340.5,
                "model": "gemini-2.5-flash",
            }
        }


class RAGChain:
    """Orchestrate retrieval + LLM generation (RAG pattern).

    The RAGChain implements the Retrieval-Augmented Generation pattern:
    1. Retrieves relevant documents based on query similarity
    2. Formats retrieved documents into a context string
    3. Constructs a system + user prompt with the context
    4. Calls the LLM to generate a contextual answer
    5. Formats the response with timing and source metadata

    Attributes:
        retriever: Retriever instance for document search.
        llm: ChatGoogleGenerativeAI instance for LLM calls.
    """

    def __init__(self, retriever: Retriever, llm):
        """Initialize RAGChain.

        Args:
            retriever: Retriever instance for document search.
            llm: ChatGoogleGenerativeAI instance for generation.
        """
        self.retriever = retriever
        self.llm = llm
        logger.info("RAGChain initialized")

    def invoke(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.7,
        include_sources: bool = True,
    ) -> RAGResponse:
        """Generate LLM response augmented with document context.

        Retrieves documents, formats them into context, constructs prompts,
        calls the LLM, and returns a structured response with sources and timing.

        Args:
            query: User question or query string.
            top_k: Number of documents to retrieve (default 5).
            temperature: LLM creativity level 0.0-1.0 (default 0.7).
                0.0 = deterministic, 1.0 = creative.
            include_sources: Whether to include retrieved documents in response
                (default True).

        Returns:
            RAGResponse with answer, sources (if requested), and execution timing.

        Raises:
            Exception: If retrieval fails or LLM call fails. Errors are logged
                with full traceback.
        """
        start_time = time.time()
        try:
            logger.info(
                f"RAGChain.invoke: query={query[:50]}..., top_k={top_k}, "
                f"temperature={temperature}, include_sources={include_sources}"
            )

            # 1. Retrieve documents
            retrieved = self.retriever.retrieve(query=query, top_k=top_k)
            logger.debug(f"Retrieved {len(retrieved)} documents")

            # 2. Format context from retrieved documents
            context_str = self._format_context(retrieved)
            logger.debug(f"Formatted context: {len(context_str)} characters")

            # 3. Build system + user prompts
            system_prompt = (
                "You are a helpful assistant that answers questions based on "
                "provided context. If the context does not contain information "
                "needed to answer the question, say so clearly."
            )

            user_prompt = (
                f"Context documents:\n\n{context_str}\n\n"
                f"Question: {query}\n\n"
                f"Please answer the question based on the context above."
            )

            logger.debug("Constructed prompts for LLM")

            # 4. Call LLM
            llm_response = self.llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            response_text = (
                llm_response.content
                if hasattr(llm_response, "content")
                else str(llm_response)
            )
            logger.debug(f"LLM response received: {len(response_text)} chars")

            # 5. Parse and format response
            execution_time_ms = (time.time() - start_time) * 1000

            # 6. Format SourceDocuments if requested
            sources_list = None
            if include_sources and retrieved:
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

            # 7. Return RAGResponse with timing
            rag_response = RAGResponse(
                response=response_text,
                query=query,
                sources=sources_list,
                execution_time_ms=execution_time_ms,
                model="gemini-2.5-flash",
            )

            logger.info(
                f"RAGChain.invoke complete: query={query[:50]}..., "
                f"time={execution_time_ms:.0f}ms, sources={len(sources_list or [])}"
            )
            return rag_response

        except Exception as e:
            logger.error(
                f"RAGChain.invoke failed: query={query[:50]}..., error={str(e)}",
                exc_info=True,
            )
            raise

    def _format_context(self, documents: List[RetrievedDocument]) -> str:
        """Format retrieved documents into a context string.

        Args:
            documents: List of RetrievedDocument objects.

        Returns:
            Formatted context string with document references and content.
        """
        if not documents:
            return "No context documents available."

        context_lines = []
        for i, doc in enumerate(documents, start=1):
            context_lines.append(
                f"[Document {i}] {doc.filename} (relevance: {doc.similarity_score:.2%})\n"
                f"Content: {doc.text}\n"
            )

        return "\n".join(context_lines)
