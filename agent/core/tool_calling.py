"""Tool-calling agent — orchestrates the LLM tool-selection strategy.

The agent itself is a thin orchestrator: it builds the LangChain executor,
delegates the heavy lifting to focused helpers, and assembles the final
``ChatResponse``. All complex logic (token accounting, decision extraction,
quality classification, tag/metadata building) lives in
``tool_calling_components.py``.

Tools are INJECTED via constructor — the container decides which tools are
active. Tracing and feedback go through the pluggable ``ObservabilityProvider``.
"""

import time
from typing import List, Mapping, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

from config.constants import (
    TRUNCATE_CONTENT_PREVIEW,
    TRUNCATE_QUERY_LOG,
)
from config.prompts import SYSTEM_PROMPT_TOOL_CALLING as SYSTEM_PROMPT
from loggers import logger
from models import ChatResponse, SourceDocument
from models.observability.decisions import DecisionLogEntry
from models.retrieval import RetrievedDocument
from observability.decorator import trace
from observability.base import ObservabilityProvider
from retrieval.formatting import build_source_documents

from .base import Agent
from .tool_calling_components import (
    ChatModelLike,
    DecisionMetadataExtractor,
    DecisionQualityClassifier,
    DecisionRecorder,
    TokenUsageCollector,
    TracingTagBuilder,
)


# ── Module-level constants ───────────────────────────────────────────

MAX_AGENT_ITERATIONS: int = 10
EARLY_STOPPING_METHOD: str = "force"

# Public agent-type identifiers (intentionally two distinct forms, per the
# original design — see ADR notes in the git history):
# - AGENT_TYPE_DATA: data-of-record (DB column, decision log, API JSON)
# - AGENT_TYPE_TAG : human-readable LangSmith tracing tag
# Unifying these would be a public API change and is explicitly out of scope.
AGENT_TYPE_DATA: str = "tool_calling"
AGENT_TYPE_TAG: str = "tool-calling"


# ── Agent ────────────────────────────────────────────────────────────


class ToolCallingAgent(Agent):
    """LangChain tool-calling agent.

    Tools are injected at construction time by the container, so the
    strategy is fully pluggable. All observability side effects (tags,
    metadata, decision records) are routed through injected services.

    All dependencies are REQUIRED (no defaults). The container wires
    them at startup; for tests, pass a no-op recorder and the NoOp
    observability provider.

    Args:
        llm: Chat model that supports tool binding. See ``ChatModelLike``.
        tools: Tools the LLM may choose to invoke.
        artifact_store: Shared list populated by tools (e.g. ``search_documents``)
            to return retrieved sources. Drained after each invocation.
        decision_tracker: Sink for ``DecisionLogEntry`` records.
        observability: Pluggable tracing + feedback backend.
    """

    def __init__(
        self,
        llm: ChatModelLike,
        tools: List[BaseTool],
        artifact_store: List[RetrievedDocument],
        decision_tracker: DecisionRecorder,
        observability: ObservabilityProvider,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._artifact_store = artifact_store
        self._decision_tracker = decision_tracker
        self._observability = observability
        self._decision_extractor = DecisionMetadataExtractor(DecisionQualityClassifier())
        logger.info(f"ToolCallingAgent initialized with {len(tools)} tools")

    # ── Public API ──────────────────────────────────────────────────

    @trace(name="ToolCallingAgent.invoke", run_type="chain")
    def invoke(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.7,
        include_sources: bool = True,
    ) -> ChatResponse:
        """Process a user query through the tool-calling strategy.

        Args:
            query: Natural language question.
            top_k: Number of documents to retrieve (passed to search tool).
            temperature: LLM creativity (0.0-1.0).
            include_sources: Whether to include source documents in the response.

        Returns:
            ``ChatResponse`` with answer, sources, and execution metadata.
        """
        start_time = time.time()
        try:
            self._log_invocation_start(query)

            token_collector = TokenUsageCollector()
            executor = self._build_executor()
            llm_latency_ms, response_text, executor_result = self._run_executor(
                executor, query, token_collector
            )
            execution_time_ms = (time.time() - start_time) * 1000

            decision = self._decision_extractor.extract(
                run_id=self._current_run_id() or "",
                query=query,
                agent_type=AGENT_TYPE_DATA,
                model_name=self._model_name,
                executor_result=executor_result,
                execution_time_ms=execution_time_ms,
                top_k=top_k,
                temperature=temperature,
            )

            sources = self._extract_sources(include_sources)
            self._publish_observability(decision, top_k, temperature)
            self._record_decision(decision)

            self._log_invocation_end(
                query=query,
                execution_time_ms=execution_time_ms,
                llm_latency_ms=llm_latency_ms,
                sources_count=len(sources or []),
                tools_used=decision.tools_used,
            )

            return self._build_response(
                query=query,
                response_text=response_text,
                sources=sources,
                execution_time_ms=execution_time_ms,
                llm_latency_ms=llm_latency_ms,
                token_collector=token_collector,
                decision=decision,
                top_k=top_k,
                temperature=temperature,
            )
        except Exception as exc:
            logger.error(
                f"ToolCallingAgent.invoke failed: "
                f"query={query[:TRUNCATE_QUERY_LOG]}..., error={exc}",
                exc_info=True,
            )
            raise

    # ── LangChain wiring ────────────────────────────────────────────

    def _build_prompt(self) -> ChatPromptTemplate:
        """Build the agent prompt with the active tool catalogue."""
        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}" for t in self._tools
        )
        system = f"{SYSTEM_PROMPT}\n\nAVAILABLE TOOLS:\n{tool_descriptions}"
        return ChatPromptTemplate.from_messages([
            ("system", system),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

    def _build_executor(self) -> AgentExecutor:
        """Build an ``AgentExecutor`` bound to the injected tools."""
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

    def _run_executor(
        self,
        executor: AgentExecutor,
        query: str,
        token_collector: TokenUsageCollector,
    ) -> tuple[float, str, Mapping[str, object]]:
        """Invoke the executor and measure LLM latency.

        Returns:
            ``(llm_latency_ms, response_text, full_executor_result)``.
        """
        llm_start = time.time()
        result: Mapping[str, object] = executor.invoke(
            {"input": query},
            config={"callbacks": [token_collector]},
        )
        llm_latency_ms = (time.time() - llm_start) * 1000
        output = result.get("output", "")
        return llm_latency_ms, output if isinstance(output, str) else str(output), result

    # ── Response assembly ───────────────────────────────────────────

    def _build_response(
        self,
        *,
        query: str,
        response_text: str,
        sources: Optional[List[SourceDocument]],
        execution_time_ms: float,
        llm_latency_ms: float,
        token_collector: TokenUsageCollector,
        decision: DecisionLogEntry,
        top_k: int,
        temperature: float,
    ) -> ChatResponse:
        """Assemble the final ``ChatResponse`` with all metadata."""
        return ChatResponse(
            response=response_text,
            query=query,
            sources=sources,
            execution_time_ms=execution_time_ms,
            model=self._model_name,
            run_id=self._current_run_id(),
            usage_metadata=token_collector.as_dict(),
            llm_latency_ms=llm_latency_ms,
            tracing_tags=TracingTagBuilder.build_tags(
                model_name=self._model_name,
                agent_type=AGENT_TYPE_TAG,
                top_k=top_k,
                temperature=temperature,
                decision=decision,
            ),
            agent_type=decision.agent_type,
            tools_used=decision.tools_used,
            chain_length=decision.chain_length,
            decision_quality=decision.decision_quality.value,
            reasoning_summary=decision.reasoning_summary,
        )

    def _extract_sources(
        self, include_sources: bool
    ) -> Optional[List[SourceDocument]]:
        """Drain the shared artifact store into ``SourceDocument`` objects.

        The ``search_documents`` tool populates this list during invocation;
        we drain it once per request and pass the captured sources downstream.
        """
        if not include_sources:
            return None
        captured = list(self._artifact_store)
        self._artifact_store.clear()
        if not captured:
            return None
        return build_source_documents(captured, True, TRUNCATE_CONTENT_PREVIEW)

    # ── Observability side effects ──────────────────────────────────

    def _publish_observability(
        self,
        decision: DecisionLogEntry,
        top_k: int,
        temperature: float,
    ) -> None:
        """Apply tags + metadata to the active trace."""
        run_id = self._current_run_id() or ""
        self._observability.apply_tags(
            run_id,
            TracingTagBuilder.build_tags(
                model_name=self._model_name,
                agent_type=AGENT_TYPE_TAG,
                top_k=top_k,
                temperature=temperature,
                decision=decision,
            ),
        )
        self._observability.apply_metadata(
            run_id,
            TracingTagBuilder.build_metadata(decision),
        )

    def _record_decision(self, decision: DecisionLogEntry) -> None:
        """Hand the decision entry to the tracker (best-effort, never raises)."""
        try:
            self._decision_tracker.record(decision)
        except Exception as exc:
            logger.warning(f"Failed to record decision: {exc}")

    # ── Logging ─────────────────────────────────────────────────────

    def _log_invocation_start(self, query: str) -> None:
        logger.info(
            f"ToolCallingAgent.invoke: query={query[:TRUNCATE_QUERY_LOG]}..., "
            f"tools={[t.name for t in self._tools]}"
        )

    @staticmethod
    def _log_invocation_end(
        query: str,
        execution_time_ms: float,
        llm_latency_ms: float,
        sources_count: int,
        tools_used: List[str],
    ) -> None:
        logger.info(
            f"ToolCallingAgent complete: query={query[:TRUNCATE_QUERY_LOG]}..., "
            f"time={execution_time_ms:.0f}ms, llm_time={llm_latency_ms:.0f}ms, "
            f"sources={sources_count}, tools_used={tools_used}"
        )

    # ── Small helpers ──────────────────────────────────────────────

    @property
    def _model_name(self) -> str:
        """Best-effort LLM model identifier (defaults to ``"unknown"``)."""
        return getattr(self._llm, "model", "unknown")

    def _current_run_id(self) -> Optional[str]:
        """The active trace run id, or ``None`` if none is in progress."""
        return self._observability.get_current_run_id()
