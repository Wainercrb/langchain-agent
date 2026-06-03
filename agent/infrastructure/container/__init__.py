"""Composition root — thin wiring layer.

Provider instantiation is delegated to infrastructure/container/providers.py.
This module only wires instances together and exposes singletons.

Change the wiring below to swap implementations across the whole app.
"""

from supabase import create_client

from config import settings
from infrastructure.container.providers import (
    _create_agent,
    _create_alert_providers,
    _create_llm_providers,
)

# ── LLM ──────────────────────────────────────────────────────────────
from infrastructure.llm import ResilientLLMProvider

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
from infrastructure.vector_store import VectorStore

db_url = settings.supabase_url
db_key = settings.supabase_key
supabase_client = create_client(db_url, db_key)
vector_store = VectorStore(supabase_client)

# ── Feedback ─────────────────────────────────────────────────────────
from infrastructure.feedback import LangSmithFeedbackProvider

feedback_service = LangSmithFeedbackProvider()

# ── Alerts ────────────────────────────────────────────────────────────
alert_service = _create_alert_providers()

# ── Decision Tracker ─────────────────────────────────────────────────
from infrastructure.observability.decisions import DecisionTracker

decision_tracker = DecisionTracker(maxlen=10000, supabase_client=supabase_client)

# ── Agent ─────────────────────────────────────────────────────────────
agent, _ = _create_agent(
    llm_provider=llm,
    tools=[],
    decision_tracker=decision_tracker,
    vector_store=vector_store,
    embeddings=embeddings,
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
