"""Tool-calling agent implementation — uses LangChain AgentExecutor.

The LLM decides which tools to call (or none) based on the query.
Tools are INJECTED via constructor — the container decides which tools are active.
"""

import os
import time
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langsmith import traceable

from config import settings
from models import ChatResponse, SourceDocument
from services.agent.base import Agent
from services.container import logger


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on available tools.

RULES:
1. Read the user's question carefully.
2. Pick EXACTLY ONE tool that matches the question type. Do not chain tools unless necessary.
3. If no tool is needed, answer directly from your training knowledge.
4. Be concise and accurate.

TOOL SELECTION GUIDE:
- User says "find the ...", "search for ...", "look up ...", "find in the ...", "find in the api documentation", "find in the requirement documents", "find in the UIQCG documents", "what does the document say about ...", "where in the docs is ..." → Use: search_documents
- Question asks about news, weather, sports, stocks, celebrities, current events → Use: web_search
- Greeting, general questions, or anything not matching above → Answer directly, no tools

EXAMPLES:
- "Find in the api documentation who is the maintainer" → search_documents
- "Search for API documentation" → search_documents
- "Find in the requirement documents the security policy" → search_documents
- "Find in the UIQCG documents the workflow steps" → search_documents
- "Who won the World Cup 2022?" → web_search
- "Hello, how are you?" → Direct answer

When you use search_documents, the tool returns actual document content.
ALWAYS answer the user's question using that content — do not reject it just because
the document filename is unexpected or the text is brief.
Only say "I don't have that information" if the retrieved documents are completely empty.
"""


class ToolCallingAgent(Agent):
    """LangChain tool-calling agent.

    Tools are injected at construction time by the container.
    This makes the agent fully pluggable — add/remove tools without touching this file.
    """

    def __init__(
        self,
        llm,
        tools: List[BaseTool],
        artifact_store: Optional[list] = None,
        default_top_k: int = 5,
        default_latest_only: bool = True,
    ):
        self._llm = llm
        self._tools = tools
        self._artifact_store = artifact_store
        self._default_top_k = default_top_k
        self._default_latest_only = default_latest_only
        logger.info(f"ToolCallingAgent initialized with {len(tools)} tools")

    def _build_prompt(self) -> ChatPromptTemplate:
        """Build agent prompt with dynamic tool descriptions."""
        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}" for t in self._tools
        )
        system = f"{SYSTEM_PROMPT}\n\nAVAILABLE TOOLS:\n{tool_descriptions}"

        return ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

    def _build_executor(self, temperature: float) -> AgentExecutor:
        """Build AgentExecutor with injected tools."""
        prompt = self._build_prompt()
        llm_with_tools = self._llm.bind_tools(self._tools)
        agent = create_tool_calling_agent(llm_with_tools, self._tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=self._tools,
            handle_parsing_errors=True,
            verbose=False,
            max_iterations=10,
            early_stopping_method="force",
        )

    @traceable(name="ToolCallingAgent.invoke", run_type="chain")
    def invoke(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.7,
        include_sources: bool = True,
        latest_only: bool = True,
    ) -> ChatResponse:
        """Process query using tool-calling agent."""
        start_time = time.time()
        try:
            logger.info(
                f"ToolCallingAgent.invoke: query={query[:50]}..., "
                f"tools={[t.name for t in self._tools]}"
            )

            executor = self._build_executor(temperature=temperature)
            result = executor.invoke({"input": query})
            response_text = result.get("output", "")

            execution_time_ms = (time.time() - start_time) * 1000

            # Extract sources from tool side-effects
            # Tool implementations that produce retrievable docs attach them
            # to a shared artifact store on the tool instance (via closure).
            sources_list = self._extract_sources(include_sources)

            run_id = str(uuid.uuid4())

            logger.info(
                f"ToolCallingAgent complete: query={query[:50]}..., "
                f"time={execution_time_ms:.0f}ms, sources={len(sources_list or [])}"
            )

            return ChatResponse(
                response=response_text,
                query=query,
                sources=sources_list,
                execution_time_ms=execution_time_ms,
                model=getattr(self._llm, "model", "unknown"),
                run_id=run_id,
            )

        except Exception as e:
            logger.error(
                f"ToolCallingAgent.invoke failed: query={query[:50]}..., error={str(e)}",
                exc_info=True,
            )
            raise

    def _extract_sources(self, include_sources: bool) -> Optional[List[SourceDocument]]:
        """Extract sources from the shared artifact store.

        The search_documents tool populates this list during invocation.
        """
        if not include_sources or not self._artifact_store:
            return None

        all_sources = list(self._artifact_store)
        self._artifact_store.clear()  # Reset for next request

        if not all_sources:
            return None

        return [
            SourceDocument(
                document_id=doc.document_id,
                filename=doc.filename,
                similarity_score=doc.similarity_score,
                version_date=doc.version_date,
                content_preview=doc.text[:200],
                chunk_id=doc.chunk_id,
            )
            for doc in all_sources
        ]
