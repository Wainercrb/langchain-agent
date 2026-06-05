"""search_documents tool — wraps the existing Retriever as a LangChain tool.

Uses the ``@tool(response_format="content_and_artifact")`` pattern:
  - Return value is ``(serialized_text, documents)``
  - LangChain puts *serialized_text* in ``ToolMessage.content`` (what the LLM sees)
  - LangChain puts *documents* in ``ToolMessage.artifact`` (what the app uses for
    source attribution)

The agent extracts artifacts from ``ToolMessage.artifact`` after invocation,
so the tool no longer mutates a shared ``artifact_store`` list.

Factory pattern: the container calls ``create_search_documents_tool(retriever)``
to produce a tool ready for agent injection.
"""

from langchain.tools import tool
from pydantic import BaseModel, Field

from models.retrieval import RetrievedDocument
from retrieval.formatting import format_documents_as_context
from loggers import logger
from .summaries import register as register_summary


def _clean_query(raw_query: str) -> str:
    """Minimal cleanup — just strip whitespace and trailing punctuation.

    The LLM constructs the query via SearchDocumentsInput instructions,
    so aggressive noise stripping is unnecessary and can hurt search quality.
    """
    return raw_query.strip().rstrip("?¿!¡.,;:")


class SearchDocumentsInput(BaseModel):
    """Input schema for the search_documents tool."""

    query: str = Field(
        description="The CORE search query — extract what the user wants to FIND, removing conversational filler. "
        "Do NOT pass greetings, pleasantries, or meta-questions ('tell me about', 'what does X say', "
        "'can you', 'i need to know', etc.). Pass a concise, keyword-focused search that captures "
        "the semantic intent. Examples: 'authentication flow in API' (not 'what does the document say "
        "about authentication in the API'), 'security policy requirements' (not 'find in the requirement "
        "documents the security policy')."
    )
    top_k: int = Field(default=5, description="Number of documents to retrieve (1-20)")


# ── Artifact summary formatter (registered for OCP) ────────────────


def _summarize_search(artifact: object) -> str | None:
    """Build a one-line summary from a list of ``RetrievedDocument``."""
    if not isinstance(artifact, list):
        return None
    docs = [d for d in artifact if isinstance(d, RetrievedDocument)]
    if not docs:
        return None
    scores = ", ".join(f"{d.similarity_score:.0%}" for d in docs)
    return f"Found {len(docs)} document{'s' if len(docs) != 1 else ''} (score: {scores})"


register_summary("search_documents", _summarize_search)


def create_search_documents_tool(
    retriever,
    default_latest_only: bool = True,
):
    """Factory: create a search_documents tool bound to the given retriever.

    Uses ``@tool(response_format="content_and_artifact")`` so the retrieved
    documents are returned as an artifact separate from the text sent to the
    LLM. The agent reads ``ToolMessage.artifact`` to build response sources.

    Args:
        retriever: A Retriever instance (from retrieval.retriever).
        default_latest_only: Whether to only retrieve latest document versions.

    Returns:
        A LangChain tool (``StructuredTool``) ready for agent injection.
    """

    @tool(
        response_format="content_and_artifact",
        args_schema=SearchDocumentsInput,
    )
    def search_documents(query: str, top_k: int = 5) -> str:
        """Search the internal document knowledge base (ingested API docs, requirements,
        UIQCG guides, project documentation, and any uploaded files).

        USE THIS whenever the user asks about project-specific content, technical
        documentation, policies, specifications, or anything that might live in
        the project's ingested documents — regardless of phrasing. The user may ask
        in any language or style; do NOT assume you can answer from your training data
        alone. Always check the documents first if the question is about the project.

        DO NOT use for: math, time/date, weather, news, general/factual knowledge,
        sports scores, stock prices, or anything requiring current/real-time data.
        Returns matching document chunks with source filenames.
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

            serialized = format_documents_as_context(
                docs, empty_message="No relevant documents found.",
            )
            # content_and_artifact: (str_for_LLM, list_of_docs_for_app)
            return serialized, docs

        except Exception as e:
            logger.error(
                f"search_documents tool error: query={query[:50]}..., error={str(e)}",
                exc_info=True,
            )
            return f"Error retrieving documents: {str(e)}", []

    return search_documents
