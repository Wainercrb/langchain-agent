"""Composition root — thin wiring layer.

This module wires all service instances together and exposes them as singletons.
Change the wiring below to swap implementations across the whole app.
"""

from supabase import create_client

from config import settings

# ── LLM ──────────────────────────────────────────────────────────────
from core.router import MultiProviderLLM


def _configured(provider_classes: list) -> list:
    """Instantiate only providers that are properly configured."""
    return [cls() for cls in provider_classes if cls.is_configured()]


def _create_llm_providers():
    """Create and return the list of configured LLM providers."""
    from llm import (
        GoogleProvider,
        OpenAIProvider,
        OpenRouterProvider,
    )

    return _configured([GoogleProvider, OpenRouterProvider, OpenAIProvider])


_llm_providers = _create_llm_providers()

llm = MultiProviderLLM(
    providers=_llm_providers,
    failure_threshold=settings.llm_circuit_failure_threshold,
    recovery_timeout=settings.llm_circuit_recovery_timeout,
    backoff_base=settings.llm_backoff_base,
    backoff_max=settings.llm_backoff_max,
)
# ── Embeddings ───────────────────────────────────────────────────────
from embeddings import GoogleEmbeddingsWrapper

embeddings = GoogleEmbeddingsWrapper(api_key=settings.google_api_key)

# ── Vector Store ─────────────────────────────────────────────────────
from vector_store import VectorStore

db_url = settings.supabase_url
db_key = settings.supabase_key
supabase_client = create_client(db_url, db_key)
vector_store = VectorStore(supabase_client)

# ── Logger ────────────────────────────────────────────────────────────
from loggers import logger

# ── Parser Registry ──────────────────────────────────────────────────
from ingestion.parsers.parser import ParserFactory

# ── Observability ─────────────────────────────────────────────────────
def _create_observability_provider():
    """Create the observability provider based on configuration."""
    from observability import set_observability_provider
    from observability.langsmith import LangSmithObservabilityProvider

    provider = LangSmithObservabilityProvider()
    set_observability_provider(provider)
    
    return provider


observability = _create_observability_provider()

# ── Alerts ────────────────────────────────────────────────────────────
def _create_alert_providers():
    """Create and return the configured alert service."""
    from alerts import (
        DiscordAlertProvider,
        SlackAlertProvider,
    )
    from core.dispatcher import MultiAlertProvider

    providers = _configured([
        DiscordAlertProvider,
        SlackAlertProvider,
    ])

    return MultiAlertProvider(providers)


alert_service = _create_alert_providers()

# ── Decision Tracker ─────────────────────────────────────────────────
from observability.decisions import DecisionTracker

decision_tracker = DecisionTracker(vector_store=vector_store, maxlen=10000)

# ── Retriever ──────────────────────────────────────────────────────────
from retrieval.retriever import Retriever

retriever = Retriever(
    vector_store=vector_store,
    embeddings=embeddings,
    logger=logger,
)

# ── Agent ─────────────────────────────────────────────────────────────

def _create_agent(llm_provider, decision_tracker, retriever, observability):
    """Create the ToolCallingAgent with search and document tools."""
    from core.tool_calling import ToolCallingAgent

    search_artifact_store = []

    from tools import create_search_documents_tool, web_search

    agent_tools = [
        create_search_documents_tool(
            retriever=retriever,
            default_latest_only=True,
        ),
        web_search,
    ]

    return ToolCallingAgent(
        llm=llm_provider.chat_model,
        tools=agent_tools,
        artifact_store=search_artifact_store,
        decision_tracker=decision_tracker,
        observability=observability,
    )


agent = _create_agent(
    llm_provider=llm,
    decision_tracker=decision_tracker,
    retriever=retriever,
    observability=observability,
)

# ── Ingestion Pipeline ─────────────────────────────────────────────────
from pathlib import Path
from ingestion.pipeline import DocumentIngestionPipeline

pipeline = DocumentIngestionPipeline(
    embeddings=embeddings,
    vector_store=vector_store,
    processed_dir=Path("data/processed"),
    failed_dir=Path("data/failed"),
    logger=logger,
    parser_registry=ParserFactory,
)

# ── Monitoring ───────────────────────────────────────────────────────
from observability.health import MonitoringScheduler
from observability.health.checks import (
    check_database,
    check_decision_drift,
    check_embeddings_service,
    check_observability_backend,
    check_process_memory,
    check_tracing_completeness,
)


def _metrics_snapshot():
    """Return current request metrics snapshot for health checks."""
    from api.metrics_store import get_request_metrics
    return get_request_metrics().snapshot()


_monitoring_checks = [
    ("db", lambda: check_database(vector_store)),
    ("observability", lambda: check_observability_backend(observability)),
    ("embeddings", lambda: check_embeddings_service(embeddings)),
    ("tracing_completeness", lambda: check_tracing_completeness(observability, _metrics_snapshot)),
    ("memory_usage", check_process_memory),
    ("decision_drift", lambda: check_decision_drift(decision_tracker)),
]

_monitoring_scheduler = MonitoringScheduler(
    checks=_monitoring_checks,
    alert_service=alert_service,
    interval_seconds=settings.monitoring_interval_seconds,
)
