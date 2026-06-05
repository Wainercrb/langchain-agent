"""Tool-calling agent — orchestrates the LLM tool-selection strategy.

The agent itself is a thin orchestrator: it builds a ``create_agent`` graph,
delegates the heavy lifting to focused helpers, and assembles the final
``ChatResponse``. All complex logic (token accounting, decision extraction,
quality classification, tag/metadata building) lives in
``tool_calling_components.py``.

Tools are INJECTED via constructor — the container decides which tools are
active. Tracing and feedback go through the pluggable ``ObservabilityProvider``.
"""

import time
from types import SimpleNamespace
from typing import Dict, List, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

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

# Public agent-type identifiers (intentionally two distinct forms, per the
# original design — see ADR notes in the git history):
# - AGENT_TYPE_DATA: data-of-record (DB column, decision log, API JSON)
# - AGENT_TYPE_TAG : human-readable LangSmith tracing tag
# Unifying these would be a public API change and is explicitly out of scope.
AGENT_TYPE_DATA: str = "tool_calling"
AGENT_TYPE_TAG: str = "tool-calling"


# ── Module-level helpers ────────────────────────────────────────────


def _format_tool_summary(tool_name: str, artifact: object) -> str:
    """Build a concise, human-readable tool-result summary.

    Delegates to the tool's registered summarizer via ``tools.summaries``
    (Open/Closed Principle — adding a new tool never requires changing
    this function). Falls back to a generic count of list items.
    """
    from tools.summaries import summarize

    return summarize(tool_name, artifact)


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
        artifact_store: Buffer populated by the agent from `ToolMessage.artifact`
            after each invocation. Drained to build response sources.
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

            # Populate artifact store from ToolMessage.artifact
            # (content_and_artifact pattern: tool returns (text, docs) tuple;
            # docs land in ToolMessage.artifact instead of mutating shared state).
            artifact_docs = self._extract_artifacts(
                executor_result.get("messages", [])
            )
            self._artifact_store.clear()
            self._artifact_store.extend(artifact_docs)

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

    @trace(name="ToolCallingAgent.stream", run_type="chain")
    def stream(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.7,
        include_sources: bool = True,
    ):
        """Process a query and yield streaming SSE-like events.

        Yields dicts with event data. The last event has ``type: "done"``
        with the full response payload (sources, timing, model info).

        Intermediate event types:
            - ``{"type": "token", "content": str}`` — text delta.
            - ``{"type": "tool_call", "tool": str, "args": dict}`` — tool
              invocation request from the LLM.
            - ``{"type": "tool_result", "tool": str, "summary": str}`` —
              tool execution result.

        The post-stream assembly (decision extraction, observability,
        source building) is identical to ``invoke()`` — the final
        ``done`` event contains the same data a ``ChatResponse`` would.
        """
        self._log_invocation_start(query)
        start_time = time.time()
        token_collector = TokenUsageCollector()
        executor = self._build_executor()

        # Capture the run_id HERE (before any yield) because the LangSmith
        # trace context is lost once the generator suspends via yield.
        _run_id = self._current_run_id()

        last_known_content = ""
        previous_msg_count = 0
        last_event_messages: list = []
        decision: Optional[DecisionLogEntry] = None

        try:
            for event in executor.stream(
                {"messages": [{"role": "user", "content": query}]},
                config={"callbacks": [token_collector]},
                stream_mode="values",
            ):
                messages: list = event.get("messages", [])
                if not messages:
                    continue

                last_event_messages = messages
                new_msgs = messages[previous_msg_count:]

                for msg in new_msgs:
                    msg_type = getattr(msg, "type", "")

                    if msg_type == "ai":
                        content = getattr(msg, "content", "") or ""

                        # Emit tool_call events for any tool invocations
                        tool_calls = getattr(msg, "tool_calls", None) or []
                        for tc in tool_calls:
                            yield {
                                "type": "tool_call",
                                "tool": tc.get("name", ""),
                                "args": tc.get("args", {}),
                            }

                        # Emit text delta when content grows
                        if content and content != last_known_content:
                            if content.startswith(last_known_content):
                                delta = content[len(last_known_content):]
                            else:
                                delta = content
                            if delta:
                                yield {"type": "token", "content": delta}
                            last_known_content = content

                    elif msg_type == "tool":
                        # Build a concise summary from the artifact
                        # (RetrievedDocument objects) instead of the raw
                        # content text which includes prompt-injection headers.
                        tool_name = getattr(msg, "name", "")
                        artifact = getattr(msg, "artifact", None)
                        summary = _format_tool_summary(tool_name, artifact)
                        yield {
                            "type": "tool_result",
                            "tool": tool_name,
                            "summary": summary,
                        }

                previous_msg_count = len(messages)

            # ── Post-stream assembly (mirrors invoke) ────────────────
            execution_time_ms = (time.time() - start_time) * 1000
            llm_latency_ms = execution_time_ms  # best-effort; fine timing lost in stream

            response_text = self._extract_response_text(last_event_messages)
            intermediate_steps = self._extract_intermediate_steps(last_event_messages)
            executor_result = {
                "intermediate_steps": intermediate_steps,
                "output": response_text,
                "messages": last_event_messages,
            }

            artifact_docs = self._extract_artifacts(last_event_messages)
            self._artifact_store.clear()
            self._artifact_store.extend(artifact_docs)

            decision = self._decision_extractor.extract(
                run_id=_run_id or "",
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

            self._log_invocation_end(
                query=query,
                execution_time_ms=execution_time_ms,
                llm_latency_ms=llm_latency_ms,
                sources_count=len(sources or []),
                tools_used=decision.tools_used,
            )

            yield {
                "type": "done",
                "response": response_text,
                "query": query,
                "sources": sources,
                "execution_time_ms": execution_time_ms,
                "llm_latency_ms": llm_latency_ms,
                "model": self._model_name,
                "run_id": _run_id,
                "usage_metadata": token_collector.as_dict(),
                "agent_type": decision.agent_type,
                "tools_used": decision.tools_used,
                "chain_length": decision.chain_length,
                "decision_quality": decision.decision_quality.value,
                "reasoning_summary": decision.reasoning_summary,
            }

        except GeneratorExit:
            # GeneratorExit (client disconnect / LangSmith cleanup) must
            # re-raise, but we record whatever decision we have first.
            if decision:
                self._record_decision(decision)
            raise

        except Exception as exc:
            logger.error(
                f"ToolCallingAgent.stream failed: "
                f"query={query[:TRUNCATE_QUERY_LOG]}..., error={exc}",
                exc_info=True,
            )
            yield {
                "type": "error",
                "message": f"Streaming failed: {exc}",
            }
        else:
            # Normal completion — record decision (outside except/finally
            # so GeneratorExit in yield above doesn't skip it).
            if decision:
                self._record_decision(decision)

    # ── LangChain wiring ────────────────────────────────────────────

    def _build_executor(self):
        """Build a ``create_agent`` graph bound to the injected tools.

        Uses the new LangChain ``create_agent()`` API (LangGraph under the
        hood) instead of the legacy ``AgentExecutor``. The graph handles
        the tool-calling loop internally — call ``.invoke()`` or ``.stream()``
        with ``{"messages": [...]}``.

        The system prompt does NOT include a tool listing because
        ``create_agent`` passes tool schemas through the model's native
        function-calling API, which is more reliable and saves tokens.
        """
        return create_agent(
            model=self._llm,
            tools=self._tools,
            system_prompt=SYSTEM_PROMPT,
        )

    def _run_executor(
        self,
        executor,
        query: str,
        token_collector: TokenUsageCollector,
    ) -> tuple[float, str, Dict[str, object]]:
        """Invoke the agent graph and measure LLM latency.

        The new ``create_agent()`` API returns ``{"messages": [...]}``.
        We extract:
        - The **response text** from the last ``AIMessage`` with content.
        - **Intermediate steps** from AI/Tool message pairs (for the
          decision recorder, which still expects the old tuple format).

        Returns:
            ``(llm_latency_ms, response_text, normalized_result)``.
        """
        llm_start = time.time()
        result: Dict[str, object] = executor.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"callbacks": [token_collector]},
        )
        llm_latency_ms = (time.time() - llm_start) * 1000

        messages = result.get("messages", [])
        response_text = self._extract_response_text(messages)
        intermediate_steps = self._extract_intermediate_steps(messages)

        # Normalise to the shape the rest of the agent expects
        # (DecisionMetadataExtractor reads executor_result["intermediate_steps"]).
        return llm_latency_ms, response_text, {
            "intermediate_steps": intermediate_steps,
            "output": response_text,
            "messages": messages,
        }

    # ── New API result helpers ──────────────────────────────────────

    @staticmethod
    def _extract_response_text(messages: List[object]) -> str:
        """Return the text content of the **last** ``AIMessage``.

        Skips AI messages that only contain tool calls (empty content).
        """
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                raw = msg.content
                return raw if isinstance(raw, str) else str(raw)
        return ""

    @staticmethod
    def _extract_intermediate_steps(
        messages: List[object],
    ) -> List[tuple[object, str]]:
        """Extract ``(action, observation)`` tuples from the message list.

        Walks the message list and pairs each ``AIMessage.tool_calls``
        with its corresponding ``ToolMessage``. Returns tuples that
        mimic the old ``AgentExecutor.intermediate_steps`` format so
        ``DecisionMetadataExtractor._parse_steps`` works unchanged.

        Returns:
            List of ``(SimpleNamespace, str)`` tuples where the namespace
            has ``tool``, ``tool_input``, and ``log`` attributes.
        """
        steps: List[tuple[object, str]] = []
        for i, msg in enumerate(messages):
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                continue

            for tc in tool_calls:
                # Locate the matching ToolMessage by tool_call_id
                observation = ""
                tc_id = tc.get("id", "") if isinstance(tc, dict) else ""
                for j in range(i + 1, len(messages)):
                    other = messages[j]
                    other_id = getattr(other, "tool_call_id", None)
                    if other_id and str(other_id) == tc_id:
                        observation = getattr(other, "content", "") or ""
                        break

                action = SimpleNamespace(
                    tool=tc["name"],
                    tool_input=tc.get("args", {}),
                    log=getattr(msg, "content", None) or "",
                )
                steps.append((action, observation))
        return steps

    @staticmethod
    def _extract_artifacts(
        messages: List[object],
    ) -> List[RetrievedDocument]:
        """Extract document artifacts from ``ToolMessage.artifact`` fields.

        The ``@tool(response_format="content_and_artifact")`` pattern stores
        retrieved documents in ``ToolMessage.artifact`` (separate from the
        text content sent to the LLM). This method collects them so the
        agent can build response sources without a shared mutable list.

        Returns:
            Flattened list of ``RetrievedDocument`` objects found across
            all ``ToolMessage.artifact`` entries.
        """
        artifacts: List[RetrievedDocument] = []
        for msg in messages:
            if getattr(msg, "type", None) != "tool":
                continue
            artifact = getattr(msg, "artifact", None)
            if artifact is None:
                continue
            if isinstance(artifact, list):
                for item in artifact:
                    if isinstance(item, RetrievedDocument):
                        artifacts.append(item)
            elif isinstance(artifact, RetrievedDocument):
                artifacts.append(artifact)
        return artifacts

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
        """Drain the artifact store into ``SourceDocument`` objects.

        The store is populated by ``_extract_artifacts`` after invocation
        (from ``ToolMessage.artifact`` fields). We drain it once per
        request and pass the captured sources downstream.
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
