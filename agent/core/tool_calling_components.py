"""Helper components for ``ToolCallingAgent``.

This module hosts the small, focused building blocks that the
``ToolCallingAgent`` orchestrates. Keeping them out of the main agent
file preserves the Single Responsibility Principle: the agent only
knows *how to compose* them, while each component knows *how to do one
thing well*.

Public components (all importable from ``agent.core.tool_calling_components``):

    ChatModelLike              — Protocol: the chat model contract the agent needs
    TokenUsageCollector       — captures token usage from LLM callbacks
    ToolCallStep              — value object: one tool invocation in a chain
    DecisionQualityClassifier — strategy: maps a tool chain → decision quality
    DecisionMetadataExtractor — turns AgentExecutor output → DecisionLogEntry
    TracingTagBuilder         — builds LangSmith trace tags + metadata
    DecisionRecorder          — Protocol (DIP): sink for DecisionLogEntry

Why split these out of ``tool_calling.py``?

    * Each helper becomes a single, easily-readable unit.
    * The agent's ``invoke()`` shrinks to a clear sequence of steps.
    * The classifier can be swapped (OCP) without touching the agent.
    * The agent depends on the ``DecisionRecorder`` Protocol, not on
      the concrete ``DecisionTracker`` (DIP).
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, runtime_checkable

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from config.constants import (
    TRUNCATE_OUTPUT_SUMMARY,
    TRUNCATE_QUERY_PREVIEW,
)
from loggers import logger
from models.observability.decisions import (
    DecisionLogEntry,
    DecisionQuality,
    ToolCallRecord,
)
from observability.decisions import DecisionTracker, compute_query_hash


__all__ = [
    "ChatModelLike",
    "DecisionMetadataExtractor",
    "DecisionQualityClassifier",
    "DecisionRecorder",
    "TokenUsageCollector",
    "ToolCallStep",
    "TracingTagBuilder",
]


# ── Chat model contract ──────────────────────────────────────────────


class ChatModelLike(Protocol):
    """Structural type for chat models usable by ``ToolCallingAgent``.

    The agent only needs ``bind_tools`` to build the LangChain executor;
    it also reads the ``model`` attribute as a best-effort identifier.
    Implemented by LangChain ``BaseChatModel`` subclasses and the
    project's custom ``MultiProviderChatModel`` in ``core/router.py``.
    """

    model: str

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
    ) -> Runnable:
        """Bind the given tools to the model and return a Runnable."""
        ...


# ── Token usage ──────────────────────────────────────────────────────


class TokenUsageCollector(BaseCallbackHandler):
    """Captures LLM token usage from callback events.

    LangChain / provider responses come in three different shapes
    (``usage_metadata``, ``usage``, ``llm_output.token_usage``). This
    handler accepts any of them and accumulates input / output tokens
    across one or more ``on_llm_end`` events.

    Usage::

        collector = TokenUsageCollector()
        executor.invoke(input, config={"callbacks": [collector]})
        collector.as_dict()  # → {"input_tokens": ..., ...} or None
    """

    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def on_llm_end(self, response: object, **kwargs: object) -> None:
        """Accumulate token counts from the LLM response (any known shape)."""
        usage = self._extract_usage(response)
        if usage is None:
            return
        self.input_tokens += self._input_tokens_of(usage)
        self.output_tokens += self._output_tokens_of(usage)

    def as_dict(self) -> Optional[Dict[str, int]]:
        """Return accumulated token counts, or ``None`` if nothing was captured."""
        if self.input_tokens == 0 and self.output_tokens == 0:
            return None
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
        }

    # ── Internal: defensive extraction ───────────────────────────────

    @staticmethod
    def _extract_usage(response: object) -> Optional[object]:
        """Pull the usage object out of any known LLM response shape."""
        return (
            getattr(response, "usage_metadata", None)
            or getattr(response, "usage", None)
            or TokenUsageCollector._from_llm_output(response)
        )

    @staticmethod
    def _from_llm_output(response: object) -> Optional[object]:
        """Last-resort extraction: legacy ``llm_output.token_usage`` dict."""
        llm_output = getattr(response, "llm_output", None) or {}
        return llm_output.get("token_usage") if isinstance(llm_output, dict) else None

    @staticmethod
    def _input_tokens_of(usage: object) -> int:
        if isinstance(usage, dict):
            return int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
        return int(getattr(usage, "input_tokens", 0) or 0)

    @staticmethod
    def _output_tokens_of(usage: object) -> int:
        if isinstance(usage, dict):
            return int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
        return int(getattr(usage, "output_tokens", 0) or 0)


# ── Tool-call steps ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolCallStep:
    """One tool invocation extracted from an ``AgentExecutor`` intermediate step.

    A normalized value object that hides whether the upstream step was
    a ``(AgentAction, observation)`` tuple or an attribute-style object.
    The decision extractor builds these; the rest of the agent only ever
    sees ``ToolCallStep`` and ``ToolCallRecord``.
    """

    tool_name: str
    tool_input: Mapping[str, object]
    output_summary: str
    rationale: Optional[str]
    order: int

    def to_record(self) -> ToolCallRecord:
        """Convert to the Pydantic model used inside ``DecisionLogEntry``."""
        return ToolCallRecord(
            tool_name=self.tool_name,
            tool_input=dict(self.tool_input),
            output_summary=self.output_summary,
            order=self.order,
        )


# ── Decision quality strategy ────────────────────────────────────────


class DecisionQualityClassifier:
    """Strategy: maps a tool-call chain to a ``DecisionQuality`` level.

    Default policy:

        * **0 tools**  → ``POOR``       (no retrieval happened)
        * **1 tool**   → ``OPTIMAL``    (the agent picked the right one)
        * **2+ tools** → ``SUBOPTIMAL`` (chained calls are usually wasteful
                                            for the current use case)

    Override the public methods to plug a different policy without
    touching the agent or the extractor (Open/Closed Principle).
    """

    def classify(
        self,
        tools_used: Sequence[str],
        chain_tools: Sequence[ToolCallStep],
    ) -> DecisionQuality:
        """Return the quality bucket for the given chain."""
        chain_length = len(tools_used)
        if chain_length == 0:
            return DecisionQuality.POOR
        if chain_length == 1:
            return DecisionQuality.OPTIMAL
        return DecisionQuality.SUBOPTIMAL

    def summarize(
        self,
        tools_used: Sequence[str],
        chain_tools: Sequence[ToolCallStep],
    ) -> str:
        """Human-readable one-liner describing the chain."""
        chain_length = len(tools_used)
        if chain_length == 0:
            return "No tool selected — direct answer or failed selection"
        if chain_length == 1:
            return f"Single tool call: {tools_used[0]}"
        return f"Chained {chain_length} tools: {', '.join(tools_used)}"

    def join_rationale(self, chain_tools: Sequence[ToolCallStep]) -> Optional[str]:
        """Join each step's rationale text into one block, or ``None``."""
        rationales = [step.rationale.strip() for step in chain_tools if step.rationale]
        return "\n---\n".join(rationales) if rationales else None


# ── Decision metadata extractor ─────────────────────────────────────


class DecisionMetadataExtractor:
    """Turns ``AgentExecutor`` output into a ``DecisionLogEntry``.

    Single responsibility: parse the executor's intermediate steps,
    classify the chain, and assemble the log entry. The agent does not
    need to know HOW the metadata is built — only THAT it is built.
    """

    def __init__(self, classifier: Optional[DecisionQualityClassifier] = None) -> None:
        self._classifier = classifier or DecisionQualityClassifier()

    def extract(
        self,
        *,
        run_id: str,
        query: str,
        agent_type: str,
        model_name: str,
        executor_result: Mapping[str, object],
        execution_time_ms: float,
        top_k: int,
        temperature: float,
    ) -> DecisionLogEntry:
        """Build a complete ``DecisionLogEntry`` for one invocation.

        Always returns an entry — even a 0-tool invocation is logged
        (with ``DecisionQuality.POOR``) so the absence of a tool is
        visible in the decision log.
        """
        chain_tools = self._parse_steps(executor_result.get("intermediate_steps", []))
        tools_used = [step.tool_name for step in chain_tools]

        logger.debug(
            f"Decision extraction: found {len(tools_used)} tools in "
            f"intermediate_steps, tools={tools_used}, "
            f"result_keys={list(executor_result.keys())}"
        )

        return DecisionLogEntry(
            run_id=run_id,
            agent_type=agent_type,
            query_preview=query[:TRUNCATE_QUERY_PREVIEW],
            query_hash=compute_query_hash(query),
            tools_used=tools_used,
            chain_length=len(tools_used),
            chain_tools=[step.to_record() for step in chain_tools],
            decision_quality=self._classifier.classify(tools_used, chain_tools),
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_used=model_name,
            top_k=top_k,
            temperature=temperature,
            latency_ms=execution_time_ms,
            reasoning_summary=self._classifier.summarize(tools_used, chain_tools),
            tool_selection_rationale=self._classifier.join_rationale(chain_tools),
        )

    # ── Step parsing ─────────────────────────────────────────────────

    @staticmethod
    def _parse_steps(intermediate_steps: Sequence[object]) -> List[ToolCallStep]:
        """Normalize every step into a ``ToolCallStep``; drop anything malformed."""
        parsed: List[ToolCallStep] = []
        for order, step in enumerate(intermediate_steps):
            normalized = DecisionMetadataExtractor._parse_step(step, order)
            if normalized is not None:
                parsed.append(normalized)
        return parsed

    @staticmethod
    def _parse_step(step: object, order: int) -> Optional[ToolCallStep]:
        """Parse one step in either of the two known upstream formats."""
        if isinstance(step, (list, tuple)) and len(step) >= 2:
            return DecisionMetadataExtractor._from_tuple_step(step, order)
        if hasattr(step, "tool"):
            return DecisionMetadataExtractor._from_attr_step(step, order)
        return None

    @staticmethod
    def _from_tuple_step(step: tuple, order: int) -> ToolCallStep:
        action, observation = step[0], step[1]
        return ToolCallStep(
            tool_name=getattr(action, "tool", None) or str(action),
            tool_input=getattr(action, "tool_input", {}) or {},
            output_summary=str(observation)[:TRUNCATE_OUTPUT_SUMMARY],
            rationale=getattr(action, "log", None),
            order=order,
        )

    @staticmethod
    def _from_attr_step(step: object, order: int) -> ToolCallStep:
        return ToolCallStep(
            tool_name=step.tool,  # type: ignore[attr-defined]
            tool_input=getattr(step, "tool_input", {}) or {},
            output_summary=str(getattr(step, "observation", ""))[:TRUNCATE_OUTPUT_SUMMARY],
            rationale=getattr(step, "log", None),
            order=order,
        )


# ── Tracing tags + metadata ──────────────────────────────────────────


class TracingTagBuilder:
    """Builds the LangSmith tags and metadata dict for one invocation.

    Pure: no I/O, no state. The agent calls ``build_tags`` and
    ``build_metadata`` and wires the results into ``ObservabilityProvider``.
    """

    @staticmethod
    def build_tags(
        *,
        model_name: str,
        agent_type: str,
        top_k: int,
        temperature: float,
        decision: DecisionLogEntry,
    ) -> List[str]:
        """Return the list of trace tags for one invocation."""
        tags = [
            f"model:{model_name}",
            f"agent:{agent_type}",
            f"top_k:{top_k}",
            f"temperature:{temperature}",
            f"decision_quality:{decision.decision_quality.value}",
            f"chain_length:{decision.chain_length}",
        ]
        if decision.tools_used:
            tags.append(f"tools_used:{','.join(decision.tools_used)}")
        return tags

    @staticmethod
    def build_metadata(decision: DecisionLogEntry) -> Dict[str, object]:
        """Return the metadata dict to attach to the active trace."""
        metadata: Dict[str, object] = {
            "agent_type": decision.agent_type,
            "decision_quality": decision.decision_quality.value,
            "chain_length": decision.chain_length,
            "tools_used": decision.tools_used,
            "reasoning_summary": decision.reasoning_summary,
            "tool_selection_rationale": decision.tool_selection_rationale,
            "query_preview": decision.query_preview,
            "latency_ms": decision.latency_ms,
            "documents_retrieved": decision.top_k,
        }
        if decision.chain_tools:
            metadata["chain_tools"] = [
                {
                    "tool": record.tool_name,
                    "order": record.order,
                    "output_summary": record.output_summary,
                }
                for record in decision.chain_tools
            ]
        return metadata


# ── Decision recorder (Dependency Inversion) ────────────────────────


@runtime_checkable
class DecisionRecorder(Protocol):
    """Anything the agent can hand a ``DecisionLogEntry`` to.

    Defined here (where it is consumed) so the agent depends on this
    Protocol, not on the concrete ``DecisionTracker``. Any class with a
    ``record(entry)`` method is a valid recorder — wire a fake in
    tests, the real tracker in production.
    """

    def record(self, entry: DecisionLogEntry) -> None:
        """Record or update a decision entry (e.g. for feedback correlation)."""
        ...
