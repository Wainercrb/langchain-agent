"""Composition root — thin wiring layer.

This module wires all service instances together and exposes them as singletons.
Change the wiring below to swap implementations across the whole app.
"""

from supabase import create_client

from config import settings

# ── LLM ──────────────────────────────────────────────────────────────
from infrastructure.llm import ResilientLLMProvider


def _configured(provider_classes: list) -> list:
    """Instantiate only providers that are properly configured."""
    return [cls() for cls in provider_classes if cls.is_configured()]


def _create_llm_providers():
    """Create and return the list of configured LLM providers."""
    from infrastructure.llm import (
        GoogleProvider,
        OpenAIProvider,
        OpenRouterProvider,
    )

    return _configured([GoogleProvider, OpenRouterProvider, OpenAIProvider])


_llm_providers = _create_llm_providers()

llm = ResilientLLMProvider(
    providers=_llm_providers,
    failure_threshold=settings.llm_circuit_failure_threshold,
    recovery_timeout=settings.llm_circuit_recovery_timeout,
    backoff_base=settings.llm_backoff_base,
    backoff_max=settings.llm_backoff_max,
)
llm.resolve_chat_model()

# ── Embeddings ───────────────────────────────────────────────────────
from infrastructure.embeddings import GoogleEmbeddingsWrapper

embeddings = GoogleEmbeddingsWrapper(api_key=settings.google_api_key)

# ── Vector Store ─────────────────────────────────────────────────────
from infrastructure.vector_stores import VectorStore

db_url = settings.supabase_url
db_key = settings.supabase_key
supabase_client = create_client(db_url, db_key)
vector_store = VectorStore(supabase_client)

# ── Logger ────────────────────────────────────────────────────────────
from infrastructure.logging import logger

# ── Tracing Orchestrator ──────────────────────────────────────────────
from infrastructure.observability.tracing import TracingOrchestratorImpl

tracing_orchestrator = TracingOrchestratorImpl()

# ── Parser Registry ──────────────────────────────────────────────────
from infrastructure.parsers.parser import ParserFactory

# ── Feedback ─────────────────────────────────────────────────────────
from infrastructure.feedback import LangSmithFeedbackProvider

feedback_service = LangSmithFeedbackProvider()

# ── Alerts ────────────────────────────────────────────────────────────

def _create_alert_providers():
    """Create and return the configured alert service."""
    from infrastructure.alerts import (
        DiscordAlertProvider,
        MultiAlertProvider,
        SlackAlertProvider,
    )

    providers = _configured([
        DiscordAlertProvider,
        SlackAlertProvider,
    ])

    return MultiAlertProvider(providers)


alert_service = _create_alert_providers()

# ── Decision Tracker ─────────────────────────────────────────────────
from infrastructure.observability.decisions import DecisionTracker

decision_tracker = DecisionTracker(maxlen=10000, supabase_client=supabase_client)

# ── Retriever ──────────────────────────────────────────────────────────
from domain.retrieval.retriever import Retriever

retriever = Retriever(
    vector_store=vector_store,
    embeddings=embeddings,
    logger=logger,
)

# ── Agent ─────────────────────────────────────────────────────────────

def _create_agent(llm_provider, decision_tracker, vector_store, embeddings, retriever):
    """Create the ToolCallingAgent with search and document tools."""
    from infrastructure.agent import ToolCallingAgent

    search_artifact_store = []

    from infrastructure.tools import create_search_documents_tool, web_search_tool

    agent_tools = [
        create_search_documents_tool(
            retriever=retriever,
            artifact_store=search_artifact_store,
            default_latest_only=True,
        ),
        web_search_tool,
    ]

    return ToolCallingAgent(
        llm=llm_provider.chat_model,
        tools=agent_tools,
        artifact_store=search_artifact_store,
        default_top_k=5,
        decision_tracker=decision_tracker,
    )


agent = _create_agent(
    llm_provider=llm,
    decision_tracker=decision_tracker,
    vector_store=vector_store,
    embeddings=embeddings,
    retriever=retriever,
)

# ── RAG Chain ──────────────────────────────────────────────────────────
from domain.chains.chain import RAGChain

rag_chain = RAGChain(
    retriever=retriever,
    llm=llm,
    decision_tracker=decision_tracker,
    logger=logger,
    tracing_orchestrator=tracing_orchestrator,
)

# ── Ingestion Pipeline ─────────────────────────────────────────────────
from pathlib import Path
from domain.ingestion.pipeline import DocumentIngestionPipeline

pipeline = DocumentIngestionPipeline(
    embeddings=embeddings,
    vector_store=vector_store,
    processed_dir=Path("data/processed"),
    failed_dir=Path("data/failed"),
    logger=logger,
    parser_registry=ParserFactory,
)

# ── Monitoring ───────────────────────────────────────────────────────
from infrastructure.observability.health import HealthVerifier, MonitoringScheduler

_health_verifier = HealthVerifier(
    vector_store=vector_store,
    embeddings=embeddings,
)

_monitoring_scheduler = MonitoringScheduler(
    health_verifier=_health_verifier,
    alert_service=alert_service,
    settings_obj=settings,
    decision_tracker=decision_tracker,
)
