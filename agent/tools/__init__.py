"""Tool factories for the tool-calling RAG agent.

Each module exposes a factory function (or ready-to-use StructuredTool).
The CONTAINER decides which tools are active and wires them into the agent.

Pattern:
    from tools import create_search_documents_tool, web_search

    # Container builds the tool list
    tools = [
        create_search_documents_tool(retriever=retriever),
        web_search,
    ]
    agent = ToolCallingAgent(llm=llm, tools=tools)
"""

from .search_documents import create_search_documents_tool
from .web_search import web_search

__all__ = [
    "create_search_documents_tool",
    "web_search",
]
