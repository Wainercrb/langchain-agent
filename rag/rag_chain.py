import logging
import time
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from api.models import SourceDocument

from .retriever import RetrievedDocument, Retriever

logger = logging.getLogger(__name__)


class RAGResponse(BaseModel):
    response: str = Field(...)
    query: str = Field(...)
    sources: Optional[List[SourceDocument]] = Field(
        default=None
    )
    execution_time_ms: float = Field(...)
    model: str = Field(
        default="gemini-2.5-flash"
    )

    model_config = ConfigDict(
        json_schema_extra={
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
    )


class RAGChain:
    def __init__(self, retriever: Retriever, llm):
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
        start_time = time.time()
        try:
            logger.info(
                f"RAGChain.invoke: query={query[:50]}..., top_k={top_k}, "
                f"temperature={temperature}, include_sources={include_sources}"
            )

            retrieved = self.retriever.retrieve(query=query, top_k=top_k)
            logger.debug(f"Retrieved {len(retrieved)} documents")

            context_str = self._format_context(retrieved)
            logger.debug(f"Formatted context: {len(context_str)} characters")

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

            llm_response = self.llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            response_text = (
                llm_response.content if hasattr(llm_response, "content") else str(llm_response)
            )
            logger.debug(f"LLM response received: {len(response_text)} chars")

            execution_time_ms = (time.time() - start_time) * 1000

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
        if not documents:
            return "No context documents available."

        context_lines = []
        for i, doc in enumerate(documents, start=1):
            context_lines.append(
                f"[Document {i}] {doc.filename} (relevance: {doc.similarity_score:.2%})\n"
                f"Content: {doc.text}\n"
            )

        return "\n".join(context_lines)
