"""Tool-calling agent implementation — uses LangChain AgentExecutor.

The LLM decides which tools to call (or none) based on the query.
Tools are INJECTED via constructor — the container decides which tools are active.
Tracing and feedback go through the pluggable ObservabilityProvider.
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Final

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

from config.constants import (
    TRUNCATE_CONTENT_PREVIEW,
    TRUNCATE_OUTPUT_SUMMARY,
    TRUNCATE_QUERY_LOG,
    TRUNCATE_QUERY_PREVIEW,
)
from config.prompts import SYSTEM_PROMPT_TOOL_CALLING as SYSTEM_PROMPT
from observability.decorator import trace
from observability.decisions import DecisionTracker
from agent.observability.base import ObservabilityProvider
from retrieval.formatting import build_source_documents
from models import ChatResponse, SourceDocument
from models.observability.decisions import DecisionLogEntry, DecisionQuality, ToolCallRecord
from .base import Agent
from loggers import logger

# Agent type has two intentionally distinct forms (per user Option 3):
# AGENT_TYPE_DATA = data-of-record (API JSON, DB column, decision log).
# AGENT_TYPE_TAG  = human-readable LangSmith tracing tag.
# Unifying these would be a public API change and is explicitly out of scope.
MAX_AGENT_ITERATIONS: Final[int] = 10
EARLY_STOPPING_METHOD: Final[str] = "force"
AGENT_TYPE_DATA: Final[str] = "tool_calling"
AGENT_TYPE_TAG: Final[str] = "tool-calling"


class _TokenCallback(BaseCallbackHandler):
    """Captures LLM token usage from callback events."""

    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Extract token counts from the LLM response."""
        usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
        if not usage:
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
        decision_tracker: Optional[Any] = None,
        observability: Optional[ObservabilityProvider] = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._artifact_store = artifact_store
        self._decision_tracker = decision_tracker
        self._observability = observability
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

    def _build_executor(self) -> AgentExecutor:
        """Build AgentExecutor with injected tools."""
        prompt = self._build_prompt()
        llm_with_tools = self._llm.bind_tools(self._tools)
        agent = create_tool_calling_agent(llm_with_tools, self._tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=self._tools,
            handle_parsing_errors=True,
            verbose=False,
            max_iterations=MAX_AGENT_ITERATIONS,
            early_stopping_method=EARLY_STOPPING_METHOD,
            return_intermediate_steps=True,
        )

    @trace(name="ToolCallingAgent.invoke", run_type="chain")
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
                f"ToolCallingAgent.invoke: query={query[:TRUNCATE_QUERY_LOG]}..., "
                f"tools={[t.name for t in self._tools]}"
            )

            token_cb = _TokenCallback()
            executor = self._build_executor()

            llm_latency_ms, response_text, executor_result = self._execute(
                executor, query, token_cb
            )

            execution_time_ms = (time.time() - start_time) * 1000

            sources_list = self._extract_sources(include_sources)

            run_id = self._observability.get_current_run_id()

            decision_metadata = self._extract_decision_metadata(
                run_id, query, executor_result, execution_time_ms, top_k, temperature
            )

            tags = self._build_tags(
                model_name=getattr(self._llm, "model", "unknown"),
                agent_type=AGENT_TYPE_TAG,
                top_k=top_k,
                temperature=temperature,
                decision_metadata=decision_metadata,
            )
            self._observability.apply_tags(run_id, tags)
            self._apply_decision_metadata(run_id, decision_metadata)

            if self._decision_tracker and decision_metadata:
                try:
                    self._decision_tracker.record(decision_metadata)
                except Exception as e:
                    logger.warning(f"Failed to record decision: {str(e)}")

            logger.info(
                f"ToolCallingAgent complete: query={query[:TRUNCATE_QUERY_LOG]}..., "
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
                tracing_tags=tags,
                agent_type=decision_metadata.agent_type if decision_metadata else AGENT_TYPE_DATA,
                tools_used=decision_metadata.tools_used if decision_metadata else [],
                chain_length=decision_metadata.chain_length if decision_metadata else 0,
                decision_quality=decision_metadata.decision_quality.value if decision_metadata else DecisionQuality.SUBOPTIMAL.value,
                reasoning_summary=decision_metadata.reasoning_summary if decision_metadata else None,
            )

        except Exception as e:
            logger.error(
                f"ToolCallingAgent.invoke failed: query={query[:TRUNCATE_QUERY_LOG]}..., error={str(e)}",
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

    def _build_tags(
        self,
        model_name: str,
        agent_type: str,
        top_k: int,
        temperature: float,
        decision_metadata: Optional[DecisionLogEntry],
    ) -> List[str]:
        """Build the list of tracing tag strings."""
        tags = [
            f"model:{model_name}",
            f"agent:{agent_type}",
            f"top_k:{top_k}",
            f"temperature:{temperature}",
        ]

        if decision_metadata:
            tags.append(f"decision_quality:{decision_metadata.decision_quality.value}")
            tags.append(f"chain_length:{decision_metadata.chain_length}")
            if decision_metadata.tools_used:
                tags.append(f"tools_used:{','.join(decision_metadata.tools_used)}")

        return tags

    def _apply_decision_metadata(
        self, run_id: str, decision_metadata: Optional[DecisionLogEntry]
    ) -> None:
        """Attach decision metadata to the active trace."""
        if not decision_metadata or not self._observability:
            return

        metadata = {
            "agent_type": decision_metadata.agent_type,
            "decision_quality": decision_metadata.decision_quality.value,
            "chain_length": decision_metadata.chain_length,
            "tools_used": decision_metadata.tools_used,
            "reasoning_summary": decision_metadata.reasoning_summary,
            "tool_selection_rationale": decision_metadata.tool_selection_rationale,
            "query_preview": decision_metadata.query_preview,
            "latency_ms": decision_metadata.latency_ms,
            "documents_retrieved": decision_metadata.top_k,
        }

        if decision_metadata.chain_tools:
            metadata["chain_tools"] = [
                {
                    "tool": t.tool_name,
                    "order": t.order,
                    "output_summary": t.output_summary,
                }
                for t in decision_metadata.chain_tools
            ]

        self._observability.apply_metadata(run_id, metadata)

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
            run_id: Run ID for this invocation.
            query: Original user query.
            executor_result: Full result dict from AgentExecutor.invoke().
            execution_time_ms: Total execution time.
            top_k: Number of documents retrieved.
            temperature: LLM temperature setting.

        Returns:
            DecisionLogEntry with tool selection and quality metadata.
        """
        intermediate_steps = executor_result.get("intermediate_steps", [])

        chain_tools: List[ToolCallRecord] = []
        tools_used: List[str] = []
        tool_selection_rationale: List[str] = []

        for i, step in enumerate(intermediate_steps):
            tool_name = None
            tool_input = {}
            output_summary = None
            rationale = None

            if isinstance(step, (list, tuple)) and len(step) >= 2:
                action = step[0]
                tool_name = getattr(action, "tool", None) or str(action)
                tool_input = getattr(action, "tool_input", {})
                output_summary = str(step[1])[:TRUNCATE_OUTPUT_SUMMARY]
                rationale = getattr(action, "log", None)
            elif hasattr(step, "tool"):
                tool_name = step.tool
                tool_input = getattr(step, "tool_input", {})
                output_summary = str(getattr(step, "observation", ""))[:TRUNCATE_OUTPUT_SUMMARY]
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
            agent_type=AGENT_TYPE_DATA,
            query_preview=query[:TRUNCATE_QUERY_PREVIEW],
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

        return build_source_documents(all_sources, True, TRUNCATE_CONTENT_PREVIEW)
