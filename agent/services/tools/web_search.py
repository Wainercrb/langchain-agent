"""Web search tool — uses DuckDuckGo for real-time information retrieval.

No API key required. Wraps langchain_community.utilities.DuckDuckGoSearchAPIWrapper
to return the top 3 results as a formatted string.
"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from services.container import logger


class WebSearchInput(BaseModel):
    """Input schema for the web_search tool."""

    query: str = Field(description="Search query for real-time web information")


def _format_search_results(results: list) -> str:
    """Format DuckDuckGo search results into a readable string."""
    if not results:
        return "No search results found."

    formatted = []
    for i, r in enumerate(results[:3], start=1):
        title = r.get("title", "Untitled")
        # DDGS returns "body" / "href"; legacy tests use "snippet" / "link"
        snippet = r.get("body") or r.get("snippet", "No description available")
        link = r.get("href") or r.get("link", "")
        formatted.append(
            f"[Result {i}] {title}\n  Snippet: {snippet}\n  Link: {link}\n"
        )

    return "\n".join(formatted)


def _web_search_func(query: str) -> str:
    """Search the web for current information not in your document knowledge base.

    Use this tool when the user asks:
    - Current events, news, or recent developments
    - Real-time information (weather, stock prices, sports scores)
    - Information not covered in the ingested documents
    - General knowledge questions where web accuracy matters

    Returns the top 3 results with title, snippet, and link.
    """
    logger.info(f"web_search tool called: query={query[:50]}...")
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            raw_results = ddgs.text(query, max_results=3)
            results = list(raw_results)
        logger.debug(f"web_search returned {len(results)} results")
        return _format_search_results(results)
    except ImportError as e:
        logger.error(f"web_search import error: {str(e)}")
        return (
            "Web search is not available: ddgs package not installed. "
            "Please add it to requirements.txt."
        )
    except Exception as e:
        logger.error(
            f"web_search tool error: query={query[:50]}..., error={str(e)}",
            exc_info=True,
        )
        return f"Error searching the web: {str(e)}"


web_search_tool = StructuredTool.from_function(
    func=_web_search_func,
    name="web_search",
    description=(
        "Search the web for current information. "
        "USE THIS when the user asks about: news, current events, weather, "
        "sports scores, stock prices, celebrity information, recent developments, "
        "or ANY topic that requires up-to-date information NOT found in documents. "
        "Examples: 'Who won the World Cup?', 'What is the weather today?', "
        "'Latest news about AI', 'What movies are playing?'. "
        "DO NOT use for: math, time, or questions about uploaded documents. "
        "Returns top 3 search result snippets."
    ),
    args_schema=WebSearchInput,
    return_direct=False,
)
