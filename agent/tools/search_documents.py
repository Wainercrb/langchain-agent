"""search_documents tool — wraps the existing Retriever as a LangChain StructuredTool.

Factory pattern: the container calls create_search_documents_tool() with the
retriever and an artifact_store list, then injects the resulting tool into the agent.
"""

import re
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from retrieval.formatting import format_documents_as_context
from loggers import logger

# Phrases to strip from user queries before semantic search.
# These are conversational "filler" that hurt embedding quality.
_QUERY_NOISE_PATTERNS = [
    r"^find\s+(in\s+the\s+)?",
    r"^search\s+(for\s+)?",
    r"^look\s+up\s+",
    r"^tell\s+me\s+(about\s+)?",
    r"\b(api|requirement|uiqcg)\s+(documentation|documents?|docs?)\b",
    r"^can\s+you\s+",
    r"^please\s+",
    r"^what\s+does\s+the\s+document\s+say\s+about\s+",
    r"^where\s+in\s+the\s+docs?\s+is\s+",
    r"^i\s+need\s+to\s+know\s+",
    r"\?$",  # trailing question mark
]


def _clean_query(raw_query: str) -> str:
    """Strip conversational filler so the retriever gets pure semantic intent.

    Only applies cleaning when the query looks like a raw user question
    (starts with filler words). LLM-constructed tool inputs are already
    focused and should NOT be cleaned.

    Example:
        "Find in the api documentation who is the maintainer?"
        → "who is the maintainer"
        "maintainer in api documentation"
        → "maintainer in api documentation"  (LLM-constructed, skip cleaning)
    """
    cleaned = raw_query.strip()

    # Skip cleaning for LLM-constructed queries: if it doesn't start with
    # conversational filler, the LLM already made it focused
    _FILLER_STARTERS = (
        "find ", "search ", "look up ", "tell me ", "can you ",
        "please ", "what does ", "where in ", "i need ",
    )
    if not cleaned.lower().startswith(_FILLER_STARTERS):
        return cleaned

    # Apply cleaning patterns one at a time, but never reduce to empty
    for pattern in _QUERY_NOISE_PATTERNS:
        result = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        result = re.sub(r"\s+", " ", result).strip()
        # Only apply if result is non-empty and meaningfully shorter
        if result and len(result) < len(cleaned):
            cleaned = result

    return cleaned or raw_query.strip()


class SearchDocumentsInput(BaseModel):
    """Input schema for the search_documents tool."""

    query: str = Field(
        description="The user's FULL original question. Do NOT shorten, rephrase, or extract keywords. Pass it exactly as the user wrote it."
    )
    top_k: int = Field(default=5, description="Number of documents to retrieve (1-20)")



def create_search_documents_tool(
    retriever,
    artifact_store: Optional[list] = None,
    default_latest_only: bool = True,
) -> StructuredTool:
    """Factory: create a search_documents tool bound to the given retriever.

    Args:
        retriever: A Retriever instance (from retrieval.retriever).
        artifact_store: Optional list that receives RetrievedDocument objects
            when the tool is called. The agent uses this to build ChatResponse.sources.
        default_latest_only: Whether to only retrieve latest document versions.

    Returns:
        A StructuredTool ready for agent injection.
    """

    def _search_docs(query: str, top_k: int = 5) -> str:
        """Search the ingested document knowledge base for relevant information.

        IMPORTANT: Pass the user's FULL original query. Do NOT shorten or
        rephrase it. The semantic search works best with complete questions.

        Use when the user asks to find, search, or look up information
        in uploaded documents using phrases like: 'find the ...', 'search for ...',
        'look up ...', 'find in the ...', 'find in the api documentation',
        'find in the requirement documents', 'find in the UIQCG documents'.
        Do NOT use for general knowledge, math, time, or web.
        """
        # Strip conversational noise so embedding matches actual content
        search_query = _clean_query(query)
        logger.info(
            f"search_documents tool called: raw={query[:50]}..., "
            f"cleaned={search_query[:50]}..., top_k={top_k}"
        )
        try:
            docs = retriever.retrieve(
                query=search_query,
                top_k=top_k,
                latest_only=default_latest_only,
            )
            logger.debug(f"search_documents retrieved {len(docs)} documents")

            if artifact_store is not None:
                artifact_store.clear()
                artifact_store.extend(docs)

            return format_documents_as_context(
                docs, empty_message="No relevant documents found."
            )
        except Exception as e:
            logger.error(
                f"search_documents tool error: query={query[:50]}..., error={str(e)}",
                exc_info=True,
            )
            return f"Error retrieving documents: {str(e)}"

    tool = StructuredTool.from_function(
        func=_search_docs,
        name="search_documents",
        description=(
            "Search the ingested document knowledge base. "
            "USE THIS when the user asks to find, search, or look up information "
            "in uploaded documents using phrases like: 'find the ...', 'search for ...', "
            "'look up ...', 'find in the ...', 'find in the api documentation', "
            "'find in the requirement documents', 'find in the UIQCG documents', "
            "'what does the document say about ...', 'where in the docs is ...'. "
            "DO NOT use for: math, time/date, weather, news, general knowledge, "
            "sports scores, stock prices, or anything requiring current/real-time data. "
            "Returns matching document chunks with source filenames."
        ),
        args_schema=SearchDocumentsInput,
        return_direct=False,
    )

    return tool
