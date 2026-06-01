"""Tool-calling agent implementation — uses LangChain AgentExecutor.

The LLM decides which tools to call (or none) based on the query.
Tools are INJECTED via constructor — the container decides which tools are active.
"""

import time
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langsmith import traceable
from langsmith.run_trees import _context as run_tree_context

from models import ChatResponse, SourceDocument
from infrastructure.agent.base import Agent
from infrastructure.logging import logger

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on available tools.

RULES:
1. Read the user's question carefully.
2. Pick EXACTLY ONE tool that matches the question type. Do not chain tools unless necessary.
3. If no tool is needed, answer directly from your training knowledge.
4. Be concise and accurate.

TOOL SELECTION GUIDE:
- User says "find the ...", "search for ...", "look up ...", "find in the ...", "find in the api documentation", "find in the requirement documents", "find in the UIQCG documents", "what does the document say about ...", "where in the docs is ..." → Use: search_documents
- Question asks about news, weather, sports, celebrities, current events → Use: web_search
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


class _TokenCallback(BaseCallbackHandler):
    """Captures LLM token usage from callback events."""

    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def on_llm_end(self, serialized: dict, response: Any, **kwargs: Any) -> None:
        """Extract token counts from the LLM response."""
        usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
        if not usage:
            return

        if isinstance(usage, dict):
            self.input_tokens += usage.get("input_tokens", usage.get("prompt_tokens", 0))
            self.output_tokens += usage.get("output_tokens", usage.get("completion_tokens", 0))
        elif hasattr(usage, "input_tokens"):
            self.input_tokens += usage.input_tokens or 0
            self.output_tokens += usage.output_tokens or 0

    def as_dict(self) -> Optional[Dict[str, int]]:
        """Return token counts as a dict, or None if no tokens were captured."""
        if self.input_tokens == 0 and self.output_tokens == 0:
            return None
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
        }


class ToolCallingAgent(Agent):
    """LangChain tool-calling agent.

    Tools are injected at construction time by the container.
    This makes the agent fully pluggable — add/remove tools without touching this file.
    """

    def __init__(
        self,
        llm: Any,
        tools: List[BaseTool],
        artifact_store: Optional[list] = None,
        default_top_k: int = 5,
        default_latest_only: bool = True,
    ) -> None:
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
        """Process query using tool-calling agent.

        Args:
            query: User's natural language question.
            top_k: Number of documents to retrieve (passed to search tool).
            temperature: LLM creativity (0.0-1.0).
            include_sources: Whether to include source documents in response.
            latest_only: Whether to search only latest document versions.

        Returns:
            ChatResponse with answer, sources, and metadata.
        """
        start_time = time.time()
        try:
            logger.info(
                f"ToolCallingAgent.invoke: query={query[:50]}..., "
                f"tools={[t.name for t in self._tools]}"
            )

            token_cb = _TokenCallback()
            executor = self._build_executor(temperature=temperature)

            llm_latency_ms, response_text = self._execute(
                executor, query, token_cb
            )

            execution_time_ms = (time.time() - start_time) * 1000

            sources_list = self._extract_sources(include_sources)

            run_id, langsmith_tags = self._capture_tracing_metadata(
                top_k, temperature
            )

            logger.info(
                f"ToolCallingAgent complete: query={query[:50]}..., "
                f"time={execution_time_ms:.0f}ms, llm_time={llm_latency_ms:.0f}ms, "
                f"sources={len(sources_list or [])}"
            )

            return ChatResponse(
                response=response_text,
                query=query,
                sources=sources_list,
                execution_time_ms=execution_time_ms,
                model=getattr(self._llm, "model", "unknown"),
                run_id=run_id,
                usage_metadata=token_cb.as_dict(),
                llm_latency_ms=llm_latency_ms,
                langsmith_tags=langsmith_tags,
            )

        except Exception as e:
            logger.error(
                f"ToolCallingAgent.invoke failed: query={query[:50]}..., error={str(e)}",
                exc_info=True,
            )
            raise

    def _execute(
        self, executor: AgentExecutor, query: str, token_cb: _TokenCallback
    ) -> tuple[float, str]:
        """Execute the agent and measure LLM latency.

        Returns:
            Tuple of (llm_latency_ms, response_text).
        """
        llm_start = time.time()
        result = executor.invoke({"input": query}, config={"callbacks": [token_cb]})
        llm_latency_ms = (time.time() - llm_start) * 1000
        return llm_latency_ms, result.get("output", "")

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
            f"model:{getattr(self._llm, 'model', 'unknown')}",
            "agent:tool-calling",
            f"top_k:{top_k}",
            f"temperature:{temperature}",
        ]
        current_run.add_tags(langsmith_tags)
        return run_id, langsmith_tags

    def _extract_sources(
        self, include_sources: bool
    ) -> Optional[List[SourceDocument]]:
        """Extract sources from the shared artifact store.

        The search_documents tool populates this list during invocation.
        """
        if not include_sources or not self._artifact_store:
            return None

        all_sources = list(self._artifact_store)
        self._artifact_store.clear()

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
