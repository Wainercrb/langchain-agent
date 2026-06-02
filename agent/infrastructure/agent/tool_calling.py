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
from models.decision import DecisionLogEntry, DecisionQuality, ToolCallRecord
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

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Extract token counts from the LLM response."""
        # LangChain passes an LLMResult object; usage may be in .usage or .llm_output
        usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
        if not usage:
            # Try llm_output for older LangChain versions
            llm_output = getattr(response, "llm_output", None) or {}
            usage = llm_output.get("token_usage", {}) if isinstance(llm_output, dict) else None

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
        decision_tracker: Optional[Any] = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._artifact_store = artifact_store
        self._default_top_k = default_top_k
        self._default_latest_only = default_latest_only
        self._decision_tracker = decision_tracker
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
            return_intermediate_steps=True,
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

            llm_latency_ms, response_text, executor_result = self._execute(
                executor, query, token_cb
            )

            execution_time_ms = (time.time() - start_time) * 1000

            sources_list = self._extract_sources(include_sources)

            run_id = self._get_run_id()

            decision_metadata = self._extract_decision_metadata(
                run_id, query, executor_result, execution_time_ms, top_k, temperature
            )

            run_id, langsmith_tags = self._capture_tracing_metadata(
                top_k, temperature, decision_metadata, run_id
            )

            if self._decision_tracker and decision_metadata:
                try:
                    self._decision_tracker.record(decision_metadata)
                except Exception as e:
                    logger.warning(f"Failed to record decision: {str(e)}")

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
                agent_type=decision_metadata.agent_type if decision_metadata else "tool_calling",
                tools_used=decision_metadata.tools_used if decision_metadata else [],
                chain_length=decision_metadata.chain_length if decision_metadata else 0,
                decision_quality=decision_metadata.decision_quality.value if decision_metadata else DecisionQuality.SUBOPTIMAL.value,
                reasoning_summary=decision_metadata.reasoning_summary if decision_metadata else None,
            )

        except Exception as e:
            logger.error(
                f"ToolCallingAgent.invoke failed: query={query[:50]}..., error={str(e)}",
                exc_info=True,
            )
            raise

    def _execute(
        self, executor: AgentExecutor, query: str, token_cb: _TokenCallback
    ) -> tuple[float, str, Any]:
        """Execute the agent and measure LLM latency.

        Returns:
            Tuple of (llm_latency_ms, response_text, executor_result).
        """
        llm_start = time.time()
        result = executor.invoke({"input": query}, config={"callbacks": [token_cb]})
        llm_latency_ms = (time.time() - llm_start) * 1000
        return llm_latency_ms, result.get("output", ""), result

    def _extract_decision_metadata(
        self,
        run_id: str,
        query: str,
        executor_result: Any,
        execution_time_ms: float,
        top_k: int,
        temperature: float,
    ) -> Optional[DecisionLogEntry]:
        """Extract decision metadata from AgentExecutor result.

        Args:
            run_id: LangSmith run ID for this invocation.
            query: Original user query.
            executor_result: Full result dict from AgentExecutor.invoke().
            execution_time_ms: Total execution time.
            top_k: Number of documents retrieved.
            temperature: LLM temperature setting.

        Returns:
            DecisionLogEntry with tool selection and quality metadata.
        """
        from datetime import datetime, timezone

        from infrastructure.decision_tracker import DecisionTracker

        intermediate_steps = executor_result.get("intermediate_steps", [])

        chain_tools: List[ToolCallRecord] = []
        tools_used: List[str] = []
        tool_selection_rationale: List[str] = []

        for i, step in enumerate(intermediate_steps):
            tool_name = None
            tool_input = {}
            output_summary = None
            rationale = None

            # Format 1: tuple/list of (AgentAction, output)
            if isinstance(step, (list, tuple)) and len(step) >= 2:
                action = step[0]
                tool_name = getattr(action, "tool", None) or str(action)
                tool_input = getattr(action, "tool_input", {})
                output_summary = str(step[1])[:200]
                rationale = getattr(action, "log", None)
            # Format 2: object with .tool attribute
            elif hasattr(step, "tool"):
                tool_name = step.tool
                tool_input = getattr(step, "tool_input", {})
                output_summary = str(getattr(step, "observation", ""))[:200]
                rationale = getattr(step, "log", None)

            if not tool_name:
                continue

            tools_used.append(tool_name)
            if rationale:
                tool_selection_rationale.append(rationale.strip())
            chain_tools.append(ToolCallRecord(
                tool_name=tool_name,
                tool_input=tool_input if isinstance(tool_input, dict) else {},
                output_summary=output_summary,
                order=i,
            ))

        chain_length = len(tools_used)

        logger.debug(
            f"Decision extraction: found {chain_length} tools in intermediate_steps, "
            f"tools={tools_used}, result_keys={list(executor_result.keys())}"
        )

        if chain_length == 0:
            decision_quality = DecisionQuality.POOR
            reasoning_summary = "No tool selected — direct answer or failed selection"
            rationale_text = None
        elif chain_length == 1:
            decision_quality = DecisionQuality.OPTIMAL
            reasoning_summary = f"Single tool call: {tools_used[0]}"
            rationale_text = tool_selection_rationale[0] if tool_selection_rationale else None
        else:
            decision_quality = DecisionQuality.SUBOPTIMAL
            reasoning_summary = f"Chained {chain_length} tools: {', '.join(tools_used)}"
            rationale_text = "\n---\n".join(tool_selection_rationale) if tool_selection_rationale else None

        return DecisionLogEntry(
            run_id=run_id,
            agent_type="tool_calling",
            query_preview=query[:200],
            query_hash=DecisionTracker.compute_query_hash(query),
            tools_used=tools_used,
            chain_length=chain_length,
            chain_tools=chain_tools,
            decision_quality=decision_quality,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_used=getattr(self._llm, "model", "unknown"),
            top_k=top_k,
            temperature=temperature,
            latency_ms=execution_time_ms,
            reasoning_summary=reasoning_summary,
            tool_selection_rationale=rationale_text,
        )

    def _get_run_id(self) -> str:
        """Extract or generate a run ID for this invocation.

        Returns:
            LangSmith run ID or UUID fallback.
        """
        current_run = run_tree_context.get_current_run_tree()
        return str(current_run.id) if current_run else str(uuid.uuid4())

    def _capture_tracing_metadata(
        self, top_k: int, temperature: float, decision_metadata: Optional[Any] = None, pre_run_id: Optional[str] = None
    ) -> tuple[str, Optional[List[str]]]:
        """Extract LangSmith run ID and apply dynamic tags.

        Args:
            top_k: Number of documents retrieved.
            temperature: LLM temperature setting.
            decision_metadata: Optional DecisionLogEntry for decision tags.
            pre_run_id: Optional pre-computed run_id to use.

        Returns:
            Tuple of (run_id, langsmith_tags).
        """
        current_run = run_tree_context.get_current_run_tree()
        run_id = pre_run_id or (str(current_run.id) if current_run else str(uuid.uuid4()))

        if not current_run:
            return run_id, None

        langsmith_tags = [
            f"model:{getattr(self._llm, 'model', 'unknown')}",
            "agent:tool-calling",
            f"top_k:{top_k}",
            f"temperature:{temperature}",
        ]

        if decision_metadata:
            try:
                langsmith_tags.append(f"decision_quality:{decision_metadata.decision_quality.value}")
                langsmith_tags.append(f"chain_length:{decision_metadata.chain_length}")
                if decision_metadata.tools_used:
                    langsmith_tags.append(f"tools_used:{','.join(decision_metadata.tools_used)}")

                # Standard LangSmith metadata for structured querying
                current_run.add_metadata({
                    "agent_type": decision_metadata.agent_type,
                    "decision_quality": decision_metadata.decision_quality.value,
                    "chain_length": decision_metadata.chain_length,
                    "tools_used": decision_metadata.tools_used,
                    "reasoning_summary": decision_metadata.reasoning_summary,
                    "tool_selection_rationale": getattr(decision_metadata, "tool_selection_rationale", None),
                    "query_preview": decision_metadata.query_preview,
                    "latency_ms": decision_metadata.latency_ms,
                })

                if decision_metadata.chain_tools:
                    current_run.add_metadata({
                        "chain_tools": [
                            {
                                "tool": t.tool_name,
                                "order": t.order,
                                "output_summary": t.output_summary,
                            }
                            for t in decision_metadata.chain_tools
                        ],
                    })
            except Exception:
                pass

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
