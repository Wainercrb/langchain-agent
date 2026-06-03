"""Composition root — single place to wire all pluggable service instances.

Change the instances below to swap implementations across the whole app.

Examples:
    # LLM
    llm = GeminiProvider(model="gemini-2.5-flash", ...)
    # llm = OpenAIProvider(model="gpt-4", api_key="sk-...")

    # Vector Store
    # vector_store = PineconeVectorStore(api_key="...")
    # vector_store = QdrantVectorStore(url="...")

    # Embeddings
    # embeddings = OpenAIEmbeddingsProvider(api_key="...")
"""

from config import settings

# ── LLM ──────────────────────────────────────────────────────────────
from infrastructure.llm import (
    OpenRouterProvider,
    GoogleProvider,
    OpenAIProvider,
    ResilientLLMProvider,
)

_llm_providers = []

if settings.google_api_key:
    _llm_providers.append(GoogleProvider(
        model=settings.gemini_model,
        temperature=settings.gemini_temperature,
        max_tokens=settings.gemini_max_tokens,
        api_key=settings.google_api_key,
    ))
if settings.openrouter_api_key:
    _llm_providers.append(OpenRouterProvider(
        model=settings.openrouter_model,
        temperature=settings.openrouter_temperature,
        max_tokens=settings.openrouter_max_tokens,
        api_key=settings.openrouter_api_key,
        timeout=settings.llm_timeout_seconds,
    ))
if settings.openai_api_key:
    _llm_providers.append(OpenAIProvider(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        max_tokens=settings.openai_max_tokens,
        api_key=settings.openai_api_key,
    ))

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
from supabase import create_client
from infrastructure.vector_store import VectorStore

db_direct_url = settings.supabase_direct_url
db_url = settings.supabase_url
db_key = settings.supabase_key
supabase_client = create_client(db_url, db_key)
vector_store = VectorStore(supabase_client)

# ── Feedback ─────────────────────────────────────────────────────────
from infrastructure.feedback import LangSmithFeedbackProvider

feedback_service = LangSmithFeedbackProvider()

# ── Alerts ────────────────────────────────────────────────────────────
from infrastructure.alerts import (
    DiscordAlertProvider,
    SlackAlertProvider,
    MultiAlertProvider,
)

_providers = []

if settings.discord_webhook_url:
    _providers.append(DiscordAlertProvider(
        webhook_url=settings.discord_webhook_url,
        rate_limit_per_minute=settings.alert_rate_limit_per_minute,
    ))

if settings.slack_webhook_url:
    _providers.append(SlackAlertProvider(
        webhook_url=settings.slack_webhook_url,
        rate_limit_per_minute=settings.alert_rate_limit_per_minute,
    ))

# Fall back to Discord-only if Slack is not configured (backward compatible)
if len(_providers) > 1:
    alert_service = MultiAlertProvider(_providers)
elif len(_providers) == 1:
    alert_service = _providers[0]
else:
    # No webhooks configured — use a no-op provider that logs warnings
    alert_service = DiscordAlertProvider(
        webhook_url=None,
        rate_limit_per_minute=settings.alert_rate_limit_per_minute,
    )

# ── Decision Tracker ─────────────────────────────────────────────────
from infrastructure.observability.decisions import DecisionTracker

decision_tracker = DecisionTracker(maxlen=10000, supabase_client=supabase_client)

# ── Agent ─────────────────────────────────────────────────────────────
from domain.retrieval.retriever import Retriever
from domain.core.chain import RAGChain
from infrastructure.agent import ToolCallingAgent, RAGChainAgent
from infrastructure.tools import create_search_documents_tool, web_search_tool

_retriever = Retriever(vector_store=vector_store, embeddings=embeddings)

if settings.use_tool_agent:
    _search_artifact_store: list = []
    _tools = [
        create_search_documents_tool(
            retriever=_retriever,
            artifact_store=_search_artifact_store,
            default_latest_only=True,
        ),
        web_search_tool,
    ]
    agent = ToolCallingAgent(
        llm=llm.chat_model,
        tools=_tools,
        artifact_store=_search_artifact_store,
        default_top_k=5,
        decision_tracker=decision_tracker,
    )
else:
    _chain = RAGChain(retriever=_retriever, llm=llm, decision_tracker=decision_tracker)
    agent = RAGChainAgent(chain=_chain)

# ── Monitoring ───────────────────────────────────────────────────────
# DecisionTracker is passed directly to the /v1/metrics route via
# build_metrics_snapshot; no global wiring needed.
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
