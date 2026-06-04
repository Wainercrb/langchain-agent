"""Document formatting utilities for retrieval results.

Provides two complementary formatters:
- ``format_documents_as_context`` — builds a text context for LLM prompts.
- ``build_source_documents`` — builds structured SourceDocument responses.
"""

from typing import List, Optional

from models import RetrievedDocument, SourceDocument
from loggers import logger


def format_documents_as_context(
    documents: List[RetrievedDocument],
    empty_message: str = "No context documents available.",
) -> str:
    """Build a text context string from retrieved documents for LLM prompts.

    Args:
        documents: Retrieved documents to format.
        empty_message: Fallback text when no documents are available.

    Returns:
        Formatted context string with document metadata and content.
    """
    if not documents:
        return empty_message

    context_lines = []
    for i, doc in enumerate(documents, start=1):
        context_lines.append(
            f"[Document {i}] {doc.filename} (relevance: {doc.similarity_score:.2%})\n"
            f"Content: {doc.text}\n"
        )

    return "\n".join(context_lines)


def build_source_documents(
    documents: list,
    include_sources: bool,
    content_preview_length: int = 200,
) -> Optional[List[SourceDocument]]:
    """Build a list of SourceDocument objects from retrieved documents.

    Args:
        documents: Raw document objects with document_id, filename,
            similarity_score, version_date, text, and chunk_id attributes.
        include_sources: Whether to include sources in the response.
        content_preview_length: Maximum characters for the content preview.

    Returns:
        List of SourceDocument objects, or None if sources are not requested.
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
