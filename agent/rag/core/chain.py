import time
import uuid
from datetime import datetime
from typing import List, Optional

from langchain_core.runnables.config import RunnableConfig

from config import settings
from models import ChatResponse, RetrievedDocument, SourceDocument

from ..retrieval.retriever import Retriever
from services.container import logger



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
        version_filter: Optional[datetime] = None,
    ) -> ChatResponse:
        start_time = time.time()
        try:
            logger.info(
                f"RAGChain.invoke: query={query[:50]}..., top_k={top_k}, "
                f"temperature={temperature}, include_sources={include_sources}, "
                f"version_filter={version_filter}"
            )

            retrieved = self.retriever.retrieve(
                query=query, top_k=top_k, version_filter=version_filter
            )
            logger.debug(f"Retrieved {len(retrieved)} documents")

            context_str = self._format_context(retrieved)
            logger.debug(f"Formatted context: {len(context_str)} characters")

            system_prompt = (
                "You are a helpful assistant that answers questions based on "
                "provided context. If the context does not contain information "
                "needed to answer the question, say so clearly. Always use the most recently ingested documents as context."
            )

            user_prompt = (
                f"Context documents:\n\n{context_str}\n\n"
                f"Question: {query}\n\n"
                f"Please answer the question based on the context above."
            )

            logger.debug("Constructed prompts for LLM")

            run_id = uuid.uuid4()
            llm_response = self.llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                config=RunnableConfig(run_id=run_id),
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

            rag_response = ChatResponse(
                response=response_text,
                query=query,
                sources=sources_list,
                execution_time_ms=execution_time_ms,
                model=self.llm.model,
                run_id=str(run_id),
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
